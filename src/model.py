import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class SDNN:
    def __init__(self, in_num, out_num, init_size, max_size, batch_size, scheme='A'):
        self.in_num = in_num
        self.out_num = out_num
        self.init_size = init_size
        self.max_size = max_size
        self.cur_size = init_size - 1
        self.batch_size = batch_size
        self.scheme = scheme
        self.name = 'SDNN'
        self.epoch = 0
        self.best_acc = 0
        self.best_acc_prune = 0
        self.now_acc = 0
        self.flag = 0
        self.active_index = []

    def neuronMask(self, mask, add_index, func='in', sparse=False, ratio=0.2):
        if sparse:
            num = int(mask.size(0) * ratio)
            for i in add_index:
                indices = list(range(mask.size(0)))
                np.random.shuffle(indices)
                selected = indices[:num]
                if func == 'in':
                    mask[selected, i] = 1
                elif func == 'out':
                    mask[i, selected] = 1
        else:
            for i in add_index:
                if func == 'in':
                    mask[:, i] = 1
                elif func == 'out':
                    mask[i, :] = 1
        return mask

    def structureInit(self, sparse=True, ratio=0.2):
        self.w1 = torch.randn(self.in_num, self.max_size) * 0.1
        self.m1 = torch.zeros(self.in_num, self.max_size)
        self.w2 = torch.randn(self.max_size, self.max_size) * 0.1
        self.m2 = torch.ones(self.max_size, self.max_size)
        self.w3 = torch.randn(self.max_size, self.out_num) * 0.1
        self.m3 = torch.zeros(self.max_size, self.out_num)
        self.w4 = torch.randn(self.in_num, self.out_num) * 0.1
        self.m4 = torch.ones(self.in_num, self.out_num)
        self.b1 = torch.zeros(1, self.max_size)
        self.b2 = torch.zeros(1, self.out_num)

        for p in [self.w1, self.w2, self.w3, self.w4, self.b1, self.b2]:
            p.requires_grad = True

        self.params = {'w1': self.w1, 'w2': self.w2, 'w3': self.w3, 'w4': self.w4,
                       'm1': self.m1, 'm2': self.m2, 'm3': self.m3, 'm4': self.m4}

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD([self.w1, self.w2, self.w3, self.w4, self.b1, self.b2],
                                   lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)

        self.active_index = list(range(self.init_size))
        self.m1.data = self.neuronMask(self.m1.data, self.active_index, sparse=sparse, ratio=ratio)
        self.m3.data = self.neuronMask(self.m3.data, self.active_index, func='out')
        self.b1.data = self.neuronMask(self.b1.data, self.active_index)

    def forward(self, x, retain_grad=True):
        self.hidden = torch.zeros(x.size(0), self.max_size)
        for j in self.active_index:
            self.hidden[:, j] = F.relu(
                torch.mm(self.hidden.clone(), self.w2[:, j] * self.m2[:, j].view(-1, 1)) +
                torch.mm(x, self.w1[:, j] * self.m1[:, j].view(-1, 1)) +
                self.b1[:, j]
            ).squeeze(1)
        out = torch.mm(self.hidden, self.w3 * self.m3) + torch.mm(x, self.w4 * self.m4) + self.b2
        if retain_grad:
            out.retain_grad()
        return out

    def apply_masks(self):
        self.w1.data *= self.m1.data
        self.w2.data *= self.m2.data
        self.w3.data *= self.m3.data
        self.w4.data *= self.m4.data

    def backwardGrad(self, outgrad):
        self.hidden.grad = torch.mm(outgrad, self.w3.T)
        rev_idx = list(reversed(self.active_index))
        for i, j in enumerate(rev_idx):
            for k in range(i):
                self.hidden.grad[:, j] += self.hidden.grad[:, k] * self.w2[j, k]

    def addConnection(self, percentile={'m1': 70, 'm2': 70, 'm3': 70, 'm4': 70}):
        loader = iter(self.trainloader)
        inputs, labels = next(loader)
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.backwardGrad(outputs)

        cov_mat = {
            'm1': torch.mm(inputs.T, self.hidden.grad).detach().numpy(),
            'm2': torch.mm(self.hidden.T, self.hidden.grad).detach().numpy(),
            'm3': torch.mm(self.hidden.T, outputs.grad).detach().numpy(),
            'm4': torch.mm(inputs.T, outputs.grad).detach().numpy()
        }

        for key in percentile:
            threshold = np.percentile(cov_mat[key][np.nonzero(cov_mat[key])], percentile[key])
            self.params[key].data[torch.tensor(cov_mat[key]) > threshold] = 1

        self.apply_masks()

    def pruneConnections(self, prune_ratio=0.2):
        for key in ['m1', 'm2', 'm3', 'm4']:
            weight = self.params[key.replace('m', 'w')]
            mask = self.params[key]
            active_weights = torch.abs(weight.data[mask.data == 1])
            if active_weights.numel() == 0:
                continue
            threshold = torch.quantile(active_weights, prune_ratio)
            to_prune = (torch.abs(weight.data) < threshold) & (mask.data == 1)
            mask.data[to_prune] = 0
            weight.data *= mask.data

    def cellDivision(self, num=1):
        activation = torch.sum(torch.abs(self.hidden), 0).detach().numpy()
        max_index_arr = np.flip(np.argsort(activation)[-num:], axis=0)

        for max_index in max_index_arr:
            add_index = len(self.active_index)
            if add_index >= self.max_size:
                continue
            if max_index not in self.active_index:
                continue

            self.active_index.insert(self.active_index.index(max_index), add_index)
            self.m1[:, add_index] = self.m1[:, max_index]
            self.m2[:, add_index] = self.m2[:, max_index]
            self.m3[add_index, :] = self.m3[max_index, :]

            self.w1[:, add_index] = self.w1[:, max_index] + torch.randn(self.in_num) * 0.01
            self.w2[:, add_index] = self.w2[:, max_index] + torch.randn(self.max_size) * 0.01
            self.w3[add_index, :] = self.w3[max_index, :] + torch.randn(self.out_num) * 0.01
            self.b1[:, add_index] = self.b1[:, max_index]

        self.apply_masks()

    def train(self, duration=10):
        for epoch in range(self.epoch, self.epoch + duration):
            self.epoch = epoch
            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs.requires_grad_(True)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.apply_masks()

    def loadData(self, X_train, y_train, X_test, y_test):
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.reshape(-1), dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test.reshape(-1), dtype=torch.long)

        self.trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size, shuffle=True)

        self.testloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_test, self.y_test),
            batch_size=self.batch_size, shuffle=False)

    def evaluate(self):
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                outputs = self.forward(inputs, retain_grad=False)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0
