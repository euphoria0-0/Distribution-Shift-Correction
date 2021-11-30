import numpy as np
import torchvision.transforms as T
import torchvision.datasets as D
import torch
from torch.utils.data import TensorDataset

class CIFAR10Dataset:
    def __init__(self, data_dir='./dataset/', train=True):
        self.nClass = 10
        self.nTrain = 50000
        self.nTest = 10000

        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) #[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        test_transform = T.Compose([
            # T.RandomCrop(32, padding=4),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        train_transform = test_transform

        if train:
            self.dataset = D.CIFAR10(data_dir + 'cifar10', train=True, transform=train_transform)
        else:
            self.dataset = D.CIFAR10(data_dir + 'cifar10', train=False, transform=test_transform)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        img, target = self.dataset[index]
        return img, target, index

    def __len__(self):
        return len(self.dataset)



class MNISTDataset:
    def __init__(self, data_dir='./dataset/', train=True, dim=3):
        self.nClass = 10
        self.nTrain = 60000
        self.nTest = 10000

        mean, std = [0.1307]*dim, [0.3081]*dim
        test_transform = T.Compose([
            # T.RandomCrop(32, padding=4),
            # T.RandomHorizontalFlip(),
            T.Grayscale(dim),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        train_transform = test_transform

        if train:
            self.dataset = D.MNIST(data_dir + 'mnist', train=True, transform=train_transform, download=True)
        else:
            self.dataset = D.MNIST(data_dir + 'mnist', train=False, transform=test_transform, download=True)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        img, target = self.dataset[index]
        return img, target, index

    def __len__(self):
        return len(self.dataset)



class SVHNDataset:
    def __init__(self, data_dir='./dataset/', train=True):
        self.nClass = 10
        self.nTrain = 73257
        self.nTest = 26032

        # mean, std = [0.4377, 0.4438, 0.4728], [0.198, 0.201, 0.197]
        # test_transform = T.Compose([
        #     # T.RandomCrop(32, padding=4),
        #     # T.RandomHorizontalFlip(),
        #     T.ToTensor(),
        #     T.Normalize(mean, std)
        # ])
        # train_transform = test_transform

        from scipy.io import loadmat
        if train:
            data = loadmat(data_dir + 'svhn/train_32x32.mat')
        else:
            data = loadmat(data_dir + 'svhn/test_32x32.mat')
        self.dataset = TensorDataset(torch.Tensor(data['X']).permute(3, 2, 0, 1),
                                     torch.squeeze(torch.Tensor(data['y'] - 1)))

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        img, target = self.dataset[index]
        return img, target, index

    def __len__(self):
        return len(self.dataset)



