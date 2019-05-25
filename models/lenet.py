import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class LeNet32(nn.Module):
    def __init__(self, n_classes):
        super(LeNet32, self).__init__()
        self.n_classes = n_classes

        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            View((-1, 16*5*5)),
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, n_classes))


    def forward(self, x, true_labels=None):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 0:
                activation1 = x
            if idx == 3:
                activation2 = x

        return x, activation1, activation2


    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print(layer, '---->', act.shape)



if __name__ == '__main__':
    import random
    import torch
    from torchsummary import summary

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    ### LENET32
    x = torch.FloatTensor(5, 3, 32, 32).uniform_(0, 1)
    model = LeNet32(n_classes=10)
    model.print_shape(x)
    output, *act = model(x)
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))
