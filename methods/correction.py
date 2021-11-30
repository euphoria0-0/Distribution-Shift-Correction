import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from utils.dataset import *
from utils.trainer import *
from torch.utils.data import Dataset



class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.sigmoid(self.linear(x))
        return x


class BCDataset(Dataset):
    '''BinaryClassificationDataset'''
    def __init__(self, dataset, label=True):
        #super(BCDataset, self).__init__()
        self.target = 1 if label else -1
        self.dataset = dataset

    def __getitem__(self, index):
        output = self.dataset[index]
        return output[0], self.target, index

    def __len__(self):
        return len(self.dataset)



def CovariateShiftCorrection(dataset, args):
    loader_args = {'batch_size': args.batch_size, 'pin_memory': True, 'shuffle': False}

    Cdataset = ConcatDataset([dataset['train'], dataset['test']])
    test_idxs = np.random.choice(len(dataset['train'])+len(dataset['test']), len(dataset['test']), replace=False)
    train_idxs = np.setdiff1d(range(len(dataset['train'])+len(dataset['test'])), test_idxs)

    dataset = {
        'train': Subset(Cdataset, train_idxs),
        'test': Subset(Cdataset, test_idxs)
    }

    dataloaders = {
        'train': DataLoader(BCDataset(dataset['train'], label=True), **loader_args),
        'test': DataLoader(BCDataset(dataset['test'], label=False), **loader_args)
    }

    model = LogisticRegression()
    model = model.to(args.device)

    print('>> Binary classification for reweighing')
    trainer = Trainer(model, dataloaders, args, binary=True)
    trainer.train()

    outputs = trainer.test(phase='correct', loader='train')
    weight = - torch.log((1 - outputs) / outputs)

    return weight



def LabelShiftCorrection(model, dataloaders, args):
    trainer = Trainer(model, dataloaders, args)
    trainer.train()

    outputs, label_dist, mu, C = trainer.test(phase='correct', confusion_mat=True)
    est_label_dist = torch.matmul(torch.linalg.inv(C), mu)
    weight = est_label_dist / label_dist

    return weight