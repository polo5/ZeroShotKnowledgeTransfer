import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import *
from models.selector import *


class TestDatasetWithIdx(Dataset):
    """Returns test samples with idx"""
    def __init__(self, dataset, datasets_path, train=False):
        dataset_path = os.path.join(datasets_path, dataset)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            self.dataset = datasets.CIFAR10(root=dataset_path, train=train, download=True, transform=transform_test)
        elif dataset == 'SVHN':
            mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            self.dataset = datasets.SVHN(root=dataset_path, split='train' if train else 'test', download=True,
                                         transform=transform_test)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        return x, index

    def __len__(self):
        return len(self.dataset)


#TODO: batch this code so it runs faster
class TransitionCurves(object):
    """
    Compute transition curves (p_j_A, p_j_B) as we adversarially
    step on netA to go from class i to j.
    """

    def __init__(self, args):
        self.args = args

        ## Load Networks
        self.netA = select_model(dataset=args.dataset, model_name=args.netA_architecture).to(device=args.device)
        self.netB = select_model(dataset=args.dataset, model_name=args.netB_architecture).to(device=args.device)
        checkpoint_netA = torch.load(os.path.join(args.netA_path), map_location=args.device)
        checkpoint_netB = torch.load(os.path.join(args.netB_path), map_location=args.device)
        self.load_state_from_keys(self.netA, checkpoint_netA)
        self.load_state_from_keys(self.netB, checkpoint_netB)
        self.netA.eval()
        self.netB.eval()

        ## Loss function
        self.criterion = nn.CrossEntropyLoss()

        ## Set up & Resume
        self.experiment_path = os.path.join(self.args.log_directory_path, self.args.experiment_name)
        self.save_model_path = os.path.join(self.args.save_model_path, self.args.experiment_name)
        self.indices = None

        if os.path.exists(self.experiment_path):
            checkpoint_path = os.path.join(self.experiment_path, 'indices.pth.tar')
            if os.path.isfile(checkpoint_path) and args.try_load_indices:
                checkpoint = torch.load(checkpoint_path)
                self.indices = checkpoint['indices']
                print(
                    '\n{} matching indices restored from ckpt, sum = {}\n'.format(len(self.indices), sum(self.indices)))

        else:
            os.makedirs(self.experiment_path)

        ## Get loaders
        if self.indices is None:
            self.indices = self.get_indices_where_nets_match()
            self.save_indices()
        if args.dataset == 'CIFAR10':
            self.n_classes = 10
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            dataset = datasets.CIFAR10(args.dataset_path, train=args.use_train_set, download=True,
                                       transform=transform_test)
        elif args.dataset == 'SVHN':
            self.n_classes = 10
            self.classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
            mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            dataset = datasets.SVHN(args.dataset_path, split='train' if args.use_train_set else 'test', download=True,
                                    transform=transform_test)
        else:
            raise NotImplementedError

        self.loader = DataLoader(dataset=dataset, batch_size=1,
                                 sampler=SubsetRandomSampler(self.indices),
                                 drop_last=False, num_workers=1)  # TODO: fix batch_size > 1 for faster runtime

        ## Save args
        print('\n---------')
        with open(os.path.join(self.experiment_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')

    def run(self):

        if self.args.check_test_accuracies:
            acc_netA, acc_netB = self.test()
            print('Checked test accuracies: netA = {:02.2f}% --- netB: {:02.2f}%'.format(acc_netA * 100, acc_netB * 100))

        pj_curves_netA = torch.zeros(self.args.n_matching_images, self.n_classes - 1, self.args.n_adversarial_steps)
        pj_curves_netB = torch.zeros(self.args.n_matching_images, self.n_classes - 1, self.args.n_adversarial_steps)

        for im_idx, (x, y) in enumerate(self.loader):
            print('image {}/{}'.format(im_idx, self.args.n_matching_images))
            x, y = x.to(self.args.device), y.to(self.args.device)

            for class_idx in range(self.n_classes - 1):
                x_adversarial = x.detach().clone()
                x_adversarial.requires_grad = True
                y_adversarial = (y + class_idx + 1) % self.n_classes

                for step_idx in range(self.args.n_adversarial_steps):
                    logits_netA, *_ = self.netA(x_adversarial)
                    with torch.no_grad():
                        logits_netB, *_ = self.netB(x_adversarial)
                    loss = self.criterion(logits_netA, y_adversarial)

                    self.netA.zero_grad()
                    loss.backward()
                    x_adversarial.data -= self.args.learning_rate * x_adversarial.grad.data
                    x_adversarial.grad.data.zero_()

                    with torch.no_grad():
                        pj_curves_netA[im_idx, class_idx, step_idx] = torch.softmax(logits_netA, dim=1)[
                            0, y_adversarial]
                        pj_curves_netB[im_idx, class_idx, step_idx] = torch.softmax(logits_netB, dim=1)[
                            0, y_adversarial]

        curves_dict = {'netA': pj_curves_netA.cpu().numpy(), 'netB': pj_curves_netB.cpu().numpy()}
        with open(os.path.join(self.experiment_path, 'curves.pickle'), 'wb') as handle:
            pickle.dump(curves_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def test(self):

        print('Checking test accuracy of networks...')

        if self.args.dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            dataset = datasets.CIFAR10(root=self.args.dataset_path, train=False, download=True,
                                       transform=transform_test)
        elif self.args.dataset == 'SVHN':
            mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            dataset = datasets.SVHN(root=self.args.dataset_path, split='test', download=True, transform=transform_test)
        else:
            raise NotImplementedError

        loader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=4)
        running_acc_netA, running_acc_netB = AggregateScalar(), AggregateScalar()

        with torch.no_grad():
            for idx, (x, y) in enumerate(loader):
                x, y = x.to(self.args.device), y.to(self.args.device)
                logits_netA, *_ = self.netA(x)
                logits_netB, *_ = self.netB(x)
                running_acc_netA.update(float(accuracy(logits_netA.data, y, topk=(1,))[0]), x.shape[0])
                running_acc_netB.update(float(accuracy(logits_netB.data, y, topk=(1,))[0]), x.shape[0])
                print(running_acc_netA.avg(), running_acc_netB.avg())

        return running_acc_netA.avg(), running_acc_netB.avg()

    def get_indices_where_nets_match(self):

        print('\nGetting matching indices for networks...')

        dataset = TestDatasetWithIdx(self.args.dataset, self.args.datasets_path, train=self.args.use_train_set)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
        assert self.args.n_matching_images < len(dataset)
        indices = []

        for x, index in loader:
            x = x.to(self.args.device)
            logits_netA, *_ = self.netA(x)
            logits_netB, *_ = self.netB(x)

            if torch.argmax(logits_netA) == torch.argmax(logits_netB):
                indices.append(int(index))
            if len(indices) == self.args.n_matching_images:
                break

        if len(indices) != self.args.n_matching_images:
            raise ValueError("Need networks that have the same predictions for at least {} images in test set".format(
                self.args.n_matching_images))

        print('\n{}/{} matching indices selected, sum = {}\n'.format(len(indices), len(dataset), sum(indices)))

        return indices

    def load_state_from_keys(self, net, ckpt, keys=('state_dict', 'student_state_dict')):
        for k in keys:
            if k in ckpt.keys():
                net.load_state_dict(ckpt[k])
                break

    def save_indices(self):
        torch.save({'indices': self.indices},
                   os.path.join(self.save_model_path, "indices.pth.tar"))


def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_torch_seeds(args.seed)

    solver = TransitionCurves(args)
    solver.run()


if __name__ == "__main__":
    import argparse
    import numpy as np
    from utils.helpers import str2bool

    print('Running...')

    parser = argparse.ArgumentParser(description='Welcome to the future')

    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN'])
    parser.add_argument('--netA_path', type=str, default='/home/paul/Pretrained/ZeroShot/CIFAR10/WRN-40-2_to_WRN-16-1/last.pth.tar')
    parser.add_argument('--netB_path', type=str, default='/home/paul/Pretrained/CIFAR10/WRN-40-2/last.pth.tar')
    parser.add_argument('--netA_architecture', type=str, default='WRN-16-1')
    parser.add_argument('--netB_architecture', type=str, default='WRN-40-2')
    parser.add_argument('--check_test_accuracies', type=str2bool, default=True)
    parser.add_argument('--use_train_set', type=str2bool, default=False)
    parser.add_argument('--n_matching_images', type=int, default=2)
    parser.add_argument('--try_load_indices', type=str2bool, default=False)
    parser.add_argument('--n_adversarial_steps', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--datasets_path', type=str, default="/home/paul/Datasets/Pytorch/")
    parser.add_argument('--log_directory_path', type=str, default="/home/paul/git/FewShotKT/logs/")
    parser.add_argument('--save_model_path', type=str, default="/home/paul/git/FewShotKT/logs/")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_gpu', type=str2bool, default=False, help='debug on cpu by setting false')
    args = parser.parse_args()

    args.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    args.dataset_path = os.path.join(args.datasets_path, args.dataset)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    args.experiment_name = 'TransitionCurves_{}_{}_{}_k{}_n{}_lr{}_seed{}'.format(args.dataset,
                                                                                  args.netA_architecture,
                                                                                  args.netB_architecture,
                                                                                  args.n_adversarial_steps,
                                                                                  args.n_matching_images,
                                                                                  args.learning_rate,
                                                                                  args.seed)
    args.experiment_name += '_train' if args.use_train_set else '_test'
    print('\nRunning on device: {}'.format(args.device))

    main(args)
