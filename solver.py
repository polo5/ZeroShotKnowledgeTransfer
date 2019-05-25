import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import shutil
from time import time
from utils.logger import *
from utils.datasets import *
from models.selector import *
from utils.loaders import *


class ZeroShotKTSolver(object):
    """ Main solver class to train and test the generator and student adversarially """
    def __init__(self, args):
        self.args = args

        ## Student and Teacher Nets
        self.teacher = select_model(dataset=args.dataset,
                                    model_name = args.teacher_architecture,
                                    pretrained = True,
                                    pretrained_models_path = args.pretrained_models_path).to(args.device)
        self.student = select_model(dataset=args.dataset,
                                    model_name=args.student_architecture,
                                    pretrained=False,
                                    pretrained_models_path = args.pretrained_models_path).to(args.device)
        self.teacher.eval()
        self.student.train()

        ## Loaders
        self.n_repeat_batch = args.n_generator_iter + args.n_student_iter
        self.generator = LearnableLoader(args = args, n_repeat_batch = self.n_repeat_batch).to(device=args.device)
        self.test_loader = get_test_loader(args)

        ## Optimizers & Schedulers
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=args.generator_learning_rate)
        self.scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_generator, args.total_n_pseudo_batches, last_epoch=-1)
        self.optimizer_student = optim.Adam(self.student.parameters(), lr=args.student_learning_rate)
        self.scheduler_student = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student, args.total_n_pseudo_batches, last_epoch=-1)

        ### Set up & Resume
        self.n_pseudo_batches = 0
        self.experiment_path = os.path.join(args.log_directory_path, args.experiment_name)
        self.save_model_path = os.path.join(args.save_model_path, args.experiment_name)
        self.logger = Logger(log_dir=self.experiment_path)

        if os.path.exists(self.experiment_path):
            if self.args.use_gpu:
                checkpoint_path = os.path.join(self.experiment_path, 'last.pth.tar')
                if os.path.isfile(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)
                    print('\nResuming from checkpoint file at batch iter {} with top 1 acc {}\n'.format(checkpoint['n_pseudo_batches'], checkpoint['test_acc']))
                    print('Running an extra {} iterations'.format(args.total_n_pseudo_batches - checkpoint['n_pseudo_batches']))
                    self.n_pseudo_batches = checkpoint['n_pseudo_batches']
                    self.generator.load_state_dict(checkpoint['generator_state_dict'])
                    self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
                    self.scheduler_generator.load_state_dict(checkpoint['scheduler_generator'])
                    self.student.load_state_dict(checkpoint['student_state_dict'])
                    self.optimizer_student.load_state_dict(checkpoint['optimizer_student'])
                    self.scheduler_student.load_state_dict(checkpoint['scheduler_student'])
            else:
                shutil.rmtree(self.experiment_path) # clear debug logs on cpu
                os.makedirs(self.experiment_path)
        else:
            os.makedirs(self.experiment_path)

        ## Save and Print Args
        print('\n---------')
        with open(os.path.join(self.experiment_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')

    def run(self):

        running_data_time, running_batch_time = AggregateScalar(), AggregateScalar()
        running_student_maxes_avg, running_teacher_maxes_avg = AggregateScalar(), AggregateScalar()
        running_student_total_loss, running_generator_total_loss = AggregateScalar(), AggregateScalar()
        student_maxes_distribution, student_argmaxes_distribution = [], []
        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []

        end = time()
        idx_pseudo = 0
        while self.n_pseudo_batches < self.args.total_n_pseudo_batches:
            x_pseudo = self.generator.__next__()
            running_data_time.update(time() - end)

            ## Take n_generator_iter steps on generator
            if idx_pseudo % self.n_repeat_batch < self.args.n_generator_iter:
                student_logits, *student_activations = self.student(x_pseudo)
                teacher_logits, *teacher_activations = self.teacher(x_pseudo)
                generator_total_loss = self.KT_loss_generator(student_logits, teacher_logits)

                self.optimizer_generator.zero_grad()
                generator_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
                self.optimizer_generator.step()


            ## Take n_student_iter steps on student
            elif idx_pseudo % self.n_repeat_batch < (self.args.n_generator_iter + self.args.n_student_iter):
                if idx_pseudo % self.n_repeat_batch == self.args.n_generator_iter:
                    with torch.no_grad(): #only need to calculate teacher logits once because teacher & x_pseudo fixed
                        teacher_logits, *teacher_activations = self.teacher(x_pseudo)

                student_logits, *student_activations = self.student(x_pseudo)
                student_total_loss = self.KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations)

                self.optimizer_student.zero_grad()
                student_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 5)
                self.optimizer_student.step()


            ## Last call to this batch, log metrics
            if (idx_pseudo + 1) % self.n_repeat_batch == 0:
                with torch.no_grad():
                    teacher_maxes, teacher_argmaxes = torch.max(torch.softmax(teacher_logits, dim=1), dim=1)
                    student_maxes, student_argmaxes = torch.max(torch.softmax(student_logits, dim=1), dim=1)
                    running_generator_total_loss.update(float(generator_total_loss))
                    running_student_total_loss.update(float(student_total_loss))
                    running_teacher_maxes_avg.update(float(torch.mean(teacher_maxes)))
                    running_student_maxes_avg.update(float(torch.mean(student_maxes)))
                    teacher_maxes_distribution.append(teacher_maxes)
                    teacher_argmaxes_distribution.append(teacher_argmaxes)
                    student_maxes_distribution.append(student_maxes)
                    student_argmaxes_distribution.append(student_argmaxes)


                if (self.n_pseudo_batches+1) % self.args.log_freq == 0:
                    test_acc = self.test()

                    with torch.no_grad():
                        print('\nBatch {}/{} -- Generator Loss: {:02.2f} -- Student Loss: {:02.2f} -- Test Acc: {:02.2f}%'.format(self.n_pseudo_batches, self.args.total_n_pseudo_batches, running_generator_total_loss.avg(), running_student_total_loss.avg(), test_acc*100))

                        self.logger.scalar_summary('TRAIN_PSEUDO/generator_total_loss', running_generator_total_loss.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/student_total_loss', running_student_total_loss.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/teacher_maxes_avg', running_teacher_maxes_avg.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/student_maxes_avg', running_student_maxes_avg.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/student_lr', self.scheduler_student.get_lr()[0], self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/generator_lr', self.scheduler_generator.get_lr()[0], self.n_pseudo_batches)
                        self.logger.scalar_summary('TIME/data_time_sec', running_data_time.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TIME/batch_time_sec', running_batch_time.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('EVALUATE/test_acc', test_acc*100, self.n_pseudo_batches)
                        self.logger.image_summary('RANDOM', self.generator.samples(n=9, grid=True), self.n_pseudo_batches)
                        self.logger.histo_summary('TEACHER_MAXES_DISTRIBUTION', torch.cat(teacher_maxes_distribution), self.n_pseudo_batches)
                        self.logger.histo_summary('TEACHER_ARGMAXES_DISTRIBUTION', torch.cat(teacher_argmaxes_distribution), self.n_pseudo_batches)
                        self.logger.histo_summary('STUDENT_MAXES_DISTRIBUTION', torch.cat(student_maxes_distribution), self.n_pseudo_batches)
                        self.logger.histo_summary('STUDENT_ARGMAXES_DISTRIBUTION', torch.cat(student_argmaxes_distribution), self.n_pseudo_batches)
                        self.logger.write_to_csv('train_test.csv')
                        self.logger.writer.flush()

                        running_data_time.reset(), running_batch_time.reset()
                        running_teacher_maxes_avg.reset(), running_student_maxes_avg.reset()
                        running_generator_total_loss.reset(), running_student_total_loss.reset(),
                        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
                        student_maxes_distribution, student_argmaxes_distribution = [], []

                if self.args.save_n_checkpoints > 1:
                    if (self.n_pseudo_batches+1) % int(self.args.total_n_pseudo_batches / self.args.save_n_checkpoints) == 0:
                        self.save_model(test_acc=test_acc)

                self.n_pseudo_batches += 1
                self.scheduler_student.step()
                self.scheduler_generator.step()

            idx_pseudo += 1
            running_batch_time.update(time() - end)
            end = time()

        if self.args.save_final_model:  # make sure last epoch saved
            self.save_model(test_acc=test_acc)

        return test_acc*100


    def test(self):

        self.student.eval()
        running_test_acc = AggregateScalar()

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.args.device), y.to(self.args.device)
                student_logits, *student_activations = self.student(x)
                acc = accuracy(student_logits.data, y, topk=(1,))[0]
                running_test_acc.update(float(acc), x.shape[0])

        self.student.train()
        return running_test_acc.avg()


    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()


    def divergence(self, student_logits, teacher_logits):
        divergence = F.kl_div(F.log_softmax(student_logits / self.args.KL_temperature, dim=1), F.softmax(teacher_logits / self.args.KL_temperature, dim=1))  # forward KL

        return divergence


    def KT_loss_generator(self, student_logits, teacher_logits):

        divergence_loss = self.divergence(student_logits, teacher_logits)
        total_loss = - divergence_loss

        return total_loss


    def KT_loss_student(self, student_logits, student_activations, teacher_logits, teacher_activations):

        divergence_loss = self.divergence(student_logits, teacher_logits)
        if self.args.AT_beta > 0:
            at_loss = 0
            for i in range(len(student_activations)):
                at_loss = at_loss + self.args.AT_beta * self.attention_diff(student_activations[i], teacher_activations[i])
        else:
            at_loss = 0

        total_loss = divergence_loss + at_loss

        return total_loss


    def save_model(self, test_acc):

        delete_files_from_name(self.save_model_path, "test_acc_", type='contains')
        file_name = "n_batches_{}_test_acc_{:02.2f}".format(self.n_pseudo_batches, test_acc * 100)
        with open(os.path.join(self.save_model_path, file_name), 'w+') as f:
            f.write("NA")

        torch.save({'args': self.args,
                    'n_pseudo_batches': self.n_pseudo_batches,
                    'generator_state_dict': self.generator.state_dict(),
                    'student_state_dict': self.student.state_dict(),
                    'optimizer_generator': self.optimizer_generator.state_dict(),
                    'optimizer_student': self.optimizer_student.state_dict(),
                    'scheduler_generator': self.scheduler_generator.state_dict(),
                    'scheduler_student': self.scheduler_student.state_dict(),
                    'test_acc': test_acc},
                   os.path.join(self.save_model_path, "last.pth.tar"))
        print("\nSaved model with test acc {:02.2f}%\n".format(test_acc * 100))
