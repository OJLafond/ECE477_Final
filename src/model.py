import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil


def neuron_mask(mask, add_index, func='in', sparse=False, ratio=0.2):
    if sparse:
        num = int(mask.size(0) * ratio)
        if func == 'in':
            for i in add_index:
                indices = np.random.permutation(mask.size(0))[:num]
                mask[indices, i] = 1
        elif func == 'out':
            for i in add_index:
                indices = np.random.permutation(mask.size(1))[:num]
                mask[i, indices] = 1
    else:
        if func == 'in':
            mask[:, add_index] = 1
        elif func == 'out':
            mask[add_index, :] = 1
    return mask


class SDNN(nn.Module):
    def __init__(self, in_num, out_num, init_size, max_size, batch_size, scheme='A'):
        super(SDNN, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.init_size = init_size
        self.max_size = max_size
        self.cur_size = init_size
        self.batch_size = batch_size
        self.scheme = scheme
        self.epoch = 0
        self.best_acc = 0
        self.best_acc_prune = 0
        self.active_index = list(range(init_size))
        self.flag = 0  # indicates pruning step

        # Define weights
        self.w1 = nn.Parameter(torch.randn(in_num, max_size) * 0.1)
        self.w2 = nn.Parameter(torch.randn(max_size, max_size) * 0.1)
        self.w3 = nn.Parameter(torch.randn(max_size, out_num) * 0.1)
        self.w4 = nn.Parameter(torch.randn(in_num, out_num) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(1, max_size))
        self.b2 = nn.Parameter(torch.zeros(1, out_num))

        # Define masks
        self.m1 = torch.zeros(in_num, max_size)
        self.m2 = torch.ones(max_size, max_size)
        self.m3 = torch.zeros(max_size, out_num)
        self.m4 = torch.ones(in_num, out_num)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD([self.w1, self.w2, self.w3, self.w4, self.b1, self.b2],
                                   lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)

    def forward(self, x):
        hidden = torch.zeros(x.size(0), self.max_size)
        for j in self.active_index:
            h_input = torch.mm(x, self.w1[:, j] * self.m1[:, j].to(x.device).float()) + self.b1[:, j]
            h_hidden = torch.mm(hidden, self.w2[:, j] * self.m2[:, j].to(x.device).float())
            hidden[:, j] = F.relu(h_input + h_hidden).squeeze(1)

        out = torch.mm(hidden, self.w3 * self.m3.to(x.device).float()) \
            + torch.mm(x, self.w4 * self.m4.to(x.device).float()) + self.b2
        return out

    def load_data(self, X_train, y_train, X_test, y_test):
        """Convert input numpy arrays to PyTorch DataLoader."""
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        self.trainloader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(TensorDataset(self.X_test, self.y_test), batch_size=self.batch_size, shuffle=False)

    def save_checkpoint(self, path, is_best=False):
        state = {
            'epoch': self.epoch,
            'best_acc': self.best_acc,
            'active_index': self.active_index,
            'state_dict': {
                'w1': self.w1.data,
                'w2': self.w2.data,
                'w3': self.w3.data,
                'w4': self.w4.data,
                'b1': self.b1.data,
                'b2': self.b2.data,
                'm1': self.m1,
                'm2': self.m2,
                'm3': self.m3,
                'm4': self.m4
            },
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)
        if is_best:
            shutil.copy(path, path.replace('.pth', '_best.pth'))

    def display_accuracy(self):
        """Evaluate accuracy on training and test sets."""
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.testloader:
                outputs = self.forward(x)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Test Accuracy: {acc:.4f}")
        return acc
