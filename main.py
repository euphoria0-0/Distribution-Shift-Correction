'''
Distribution Shift Correction
'''
import argparse
import os

from utils.dataset import *
from utils.trainer import *
from utils.model import *
from methods.correction import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu cuda index')

    parser.add_argument('--dataset', help='dataset', type=str, default='MNIST')
    parser.add_argument('--data_dir', help='data path', type=str, default='D:/data/img_clf/')

    parser.add_argument('--method', type=str, default=None)

    parser.add_argument('-E', '--num_epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('-B', '--batch_size', type=int, default=128, help='Batch size used for training only')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--milestone', type=list, default=[50,80], help='milestones for multistep lr scheduler')

    parser.add_argument('--fix_seed', action='store_true', default=True, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # set up
    args = get_args()
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print(f'Current cuda device: {torch.cuda.current_device()}')

    # set
    if args.fix_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True

    # set data
    if args.dataset == 'CIFAR10':
        dataset = {'train': CIFAR10Dataset(args.data_dir, True),
                   'test': CIFAR10Dataset(args.data_dir, False)}
        args.nClass, args.nTrain, args.nTest = 10, 50000, 10000
    elif args.dataset == 'MNIST':
        dataset = {'train': MNISTDataset(args.data_dir, True),
                   'test': MNISTDataset(args.data_dir, False)}
        dataset1 = {'train': MNISTDataset(args.data_dir, True, dim=1),
                    'test': MNISTDataset(args.data_dir, False, dim=1)}
        args.nClass, args.nTrain, args.nTest = 10, 60000, 10000
    tar_dataset = {'train': SVHNDataset(args.data_dir, True),
                   'test': SVHNDataset(args.data_dir, False)}
    print('Arguments:', end=' ')
    print(', '.join(f'{k}={v}' for k, v in vars(args).items()))
    print(f'Source Dataset: {args.dataset}')

    # set dataloader
    loader_args = {'batch_size': args.batch_size, 'pin_memory': True, 'shuffle': False}
    dataloaders = {
        'train': DataLoader(dataset['train'], **loader_args),
        'test': DataLoader(dataset['test'], batch_size=1000, pin_memory=True, shuffle=True)
    }
    tar_dataloaders = {
        'train': DataLoader(tar_dataset['train'], **loader_args),
        'test': DataLoader(tar_dataset['test'], batch_size=1000, pin_memory=True, shuffle=True)
    }

    # set model for cifar10
    if args.dataset == 'CIFAR10':
        model = ResNet18(pretrained=False, num_classes=10)
    elif args.dataset == 'MNIST':
        model = ResNet18(pretrained=False, num_classes=10)
    model = model.to(args.device)

    # method
    # covariate shift correction
    if args.method == 'CovariateShiftCorrection':
        weight = CovariateShiftCorrection(dataset1, args)
    elif args.method == 'LabelShiftCorrection':
        weight = LabelShiftCorrection(model, dataloaders, args)
    else:
        weight = None

    # train
    trainer = Trainer(model, dataloaders, args, weight=weight)
    train_acc = trainer.train()
    test_acc = trainer.test()
    model = trainer.get_model()

    # test for target domain
    print('Target Dataset: SVHN')
    tar_trainer = Trainer(model, tar_dataloaders, args, weight=weight)
    tar_test_acc = tar_trainer.test()

    # save model
    os.makedirs('./models/', exist_ok=True)
    torch.save(model.state_dict(), f"models/{args.dataset}_model_{args.method}.pt")
