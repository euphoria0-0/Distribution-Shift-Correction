import sys
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


class Trainer:
    def __init__(self, model, dataloaders, args, weight=None, binary=False):
        self.model = model
        self.dataloaders = dataloaders
        self.device = args.device
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.args = args
        self.binary = binary
        self.weight = weight
        if self.weight is not None:
            self.weight = self.weight.squeeze().to(self.device)

        # loss function
        if binary:
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.wdecay)
        # learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone, gamma=0.1)

    def train(self):
        self.model.train()
        best_acc = 0.0
        for epoch in range(self.num_epoch):
            train_acc = self.train_epoch(epoch)
            # validation
            '''val_acc = self.test('val')
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = deepcopy(self.model)

        self.model = best_model'''
        return train_acc #best_acc

    def train_epoch(self, epoch):
        train_loss, correct, total = 0., 0, 0

        for idx, (input, labels, _) in enumerate(self.dataloaders['train']):
            input, labels = input.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input)
            if self.binary:
                loss = self.criterion(output, labels.float().unsqueeze(dim=1))
            else:
                loss = self.criterion(output, labels.long())
            _, preds = torch.max(output.data, 1)

            if self.weight is not None:
                #print(loss[:3], self.weight[:3])
                loss *= self.weight[idx*self.args.batch_size: (idx+1)*self.args.batch_size]
            #print(loss.size(), end=' ')
            #print(loss.mean().size())

            loss.mean().backward()

            train_loss += loss.sum().item()
            correct += torch.sum(preds == labels.data).cpu().data.numpy()
            total += input.size(0)

            self.optimizer.step()

            torch.cuda.empty_cache()

        self.lr_scheduler.step()

        train_acc = correct / total
        train_loss = train_loss / total

        sys.stdout.write('\rEpoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch + 1, self.num_epoch,
                                                                                 train_loss, train_acc))
        sys.stdout.flush()
        torch.cuda.empty_cache()
        return train_acc

    def test(self, phase='test', confusion_mat=False, loader='test'):
        self.model.eval()
        labels_lst = torch.empty((0))
        nClass = 1 if self.binary else self.args.nClass
        outputs_lst = torch.empty((0, nClass))
        confusion_matrix = torch.zeros(self.args.nClass, self.args.nClass)

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            for input, labels, _ in self.dataloaders[loader]:
                input, labels = input.to(self.device), labels.to(self.device)
                output = self.model(input)
                #loss = self.criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                #test_loss += loss.item() * input.size(0)
                correct += preds.eq(labels).sum().cpu().data.numpy()
                total += input.size(0)

                if phase=='correct':
                    outputs_lst = torch.cat((outputs_lst, output.detach().cpu().data), dim=0)
                    if confusion_mat:
                        for t, p in zip(labels.view(-1), preds.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                        labels_lst = torch.cat((labels_lst, labels.detach().cpu().data), dim=0)

                torch.cuda.empty_cache()

        test_acc = correct / total
        print(' TestAcc: {:.4f}'.format(test_acc))
        torch.cuda.empty_cache()
        if phase == 'correct':
            if confusion_mat:
                return outputs_lst, torch.bincount(labels_lst)/total, outputs_lst.mean(dim=0), confusion_matrix
            else:
                return outputs_lst
        else:
            return test_acc

    def get_model(self):
        return self.model