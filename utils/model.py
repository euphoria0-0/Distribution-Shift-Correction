'''
Ref: https://github.com/y0ast/pytorch-snippets/blob/main/minimal_cifar/train_cifar.py
'''

import torchvision.models as models
import torch
import torch.nn as nn
import torch.functional as F


class ResNet18(torch.nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super().__init__()

        self.resnet = models.resnet18(pretrained=pretrained, num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        #x = F.log_softmax(x, dim=1)

        return x


class ResNet18_MNIST(torch.nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super().__init__()

        self.resnet = models.resnet18(pretrained=pretrained, num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.resnet(x)
        return x