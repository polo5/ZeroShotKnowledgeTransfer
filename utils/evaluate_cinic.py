import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from models.selector import *
from utils.datasets import *
from utils.helpers import *

## Datasets

#dataset = 'CIFAR10'
dataset = 'CINIC10'

## CIFAR10
if dataset == 'CIFAR10':
    dataset_path = '/home/paul/Datasets/Pytorch/CIFAR10'
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform)

elif dataset == 'CINIC10':
    dataset_path = '/home/paul/Datasets/Pytorch/CINIC10-ImageNet'
    mean, std = (0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)

loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=256,
    shuffle=True, drop_last=False, num_workers=2,
    pin_memory=False)
########

device = 'cpu'
student_path = '/home/paul/Pretrained/ZeroShot/CIFAR10/WRN-40-2_to_WRN-16-1/last.pth.tar'
teacher_path = '/home/paul/Pretrained/CIFAR10/WRN-40-2/last.pth.tar'

student = select_model(dataset='CIFAR10', model_name='WRN-16-1').to(device=device)
checkpoint = torch.load(student_path, map_location=device)
student.load_state_dict(checkpoint['student_state_dict'])
student.eval() # somehow student is very bad if not in eval mode for CIFAR10, but better than student for CINIC10

teacher = select_model(dataset='CIFAR10', model_name='WRN-40-2').to(device=device)
checkpoint = torch.load(teacher_path, map_location=device)
teacher.load_state_dict(checkpoint['state_dict'])
teacher.eval()

running_acc1_student, running_acc1_teacher = AggregateScalar(), AggregateScalar()

with torch.no_grad():
    for x, y in loader:
        #print(x.shape, y.shape)
        x, y = x.to(device), y.to(device)
        logits_student, *_ = student(x)
        logits_teacher, *_ = teacher(x)
        acc1_student = accuracy(logits_student.data, y, topk=(1,))[0]
        acc1_teacher = accuracy(logits_teacher.data, y, topk=(1,))[0]
        print(acc1_student, acc1_teacher)
        running_acc1_student.update(float(acc1_student), x.shape[0])
        running_acc1_teacher.update(float(acc1_teacher), x.shape[0])

        print('student acc {}, teacher acc {}'.format(running_acc1_student.avg()*100, running_acc1_teacher.avg()*100))