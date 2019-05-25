import os

from models.lenet import *
from models.wresnet import *


def select_model(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None):
    if dataset in ['SVHN', 'CIFAR10']:
        n_classes = 10
        if model_name == 'LeNet':
            model = LeNet32(n_classes=n_classes)
        elif model_name == 'WRN-16-1':
            model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0.0)
        elif model_name == 'WRN-16-2':
            model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0.0)
        elif model_name == 'WRN-40-1':
            model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0.0)
        elif model_name == 'WRN-40-2':
            model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0.0)
        else:
            raise NotImplementedError

        if pretrained:
            model_path = os.path.join(pretrained_models_path, dataset, model_name, "last.pth.tar")
            print('Loading Model from {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

    else:
        raise NotImplementedError

    return model


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import time

    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)

    t0 = time.time()
    model = select_model('CIFAR10', model_name='WRN-16-2')
    output, *act = model(x)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))
