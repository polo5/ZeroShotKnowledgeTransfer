import math

from torchvision.utils import make_grid

from models.generator import *
from utils.helpers import *


def visualize(x_norm, dataset):
    """
    This un-normalizes for visualization purposes only.
    """
    if dataset == 'SVHN':
        mean = torch.Tensor([0.4377, 0.4438, 0.4728]).view(1, 3, 1, 1).to(x_norm.device)
        std = torch.Tensor([0.1980, 0.2010, 0.1970]).view(1, 3, 1, 1).to(x_norm.device)
        x = x_norm * std + mean
    elif dataset == 'CIFAR10':
        mean = torch.Tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x_norm.device)
        std = torch.Tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x_norm.device)
        x = x_norm * std + mean
    else:
        raise NotImplementedError

    return x


class LearnableLoader(nn.Module):
    def __init__(self, args, n_repeat_batch):
        """
        Infinite loader, which contains a learnable generator.
        """

        super(LearnableLoader, self).__init__()
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.n_repeat_batch = n_repeat_batch
        self.z_dim = args.z_dim
        self.generator = Generator(args.z_dim).to(device=args.device)
        self.device = args.device

        self._running_repeat_batch_idx = 0
        self.z = torch.randn((args.batch_size, args.z_dim)).to(device=args.device)

    def __next__(self):
        if self._running_repeat_batch_idx == self.n_repeat_batch:
            self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)
            self._running_repeat_batch_idx = 0

        images = self.generator(self.z)
        self._running_repeat_batch_idx += 1
        return images

    def samples(self, n, grid=True):
        """
        :return: if grid returns single grid image, else
        returns n images.
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.z_dim)).to(device=self.device)
            images = visualize(self.generator(z), dataset=self.dataset).cpu()
            if grid:
                images = make_grid(images, nrow=round(math.sqrt(n)), normalize=True)

        self.generator.train()
        return images

    def __iter__(self):
        return self


if __name__ == '__main__':

    ## Loader
    args = type('', (), {})()  # dummy object
    args.dataset = 'CIFAR10'
    args.batch_size = 128
    args.z_dim = 100
    args.device = 'cpu'
    loader = LearnableLoader(args, n_repeat_batch=5)

    ## Check n_repeat works while iterating
    for idx, x in enumerate(loader):
        print(x.shape, float(torch.sum(x)))
        if (idx + 1) == 10:
            break

    grid = loader.samples(n=9)
    plot_image(grid)
