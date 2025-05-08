### Find Optimal Architecture using SCANN ###

import random
import torch

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import shutil

# define some util functions

def neuronMask(mask, add_index, func='in', sparse=False, ratio=0.2):
    if sparse:
        num = int(mask.size(0)*ratio)
        if func == 'in':
            for i in add_index:
                indices = list(range(mask.size(0)))
                np.random.shuffle(indices)
                for j in indices[:num]:
                    mask[j, i] = 1
        elif func == 'out':
            for i in add_index:
                indices = list(range(mask.size(1)))
                np.random.shuffle(indices)
                for j in indices[:num]:
                    mask[i, j] = 1
    else:
        if func == 'in':
            for i in add_index:
	            mask[:, i] = 1
        elif func == 'out':
            for i in add_index:
                mask[i, :] = 1
    return mask

class SDNN(object):
    def __init__(self, in_num, out_num, init_size, max_size, batch_size, scheme= 'A', **kwargs):
        self.in_num = in_num # The number of input features
        self.out_num = out_num # The number of output classes
        self.init_size = init_size # the inital number of hidden neurons
        self.max_size = max_size  # max number of allowed hidden neurons in the architecture
        self.cur_size = init_size - 1  # current size of hidden neurons
        self.batch_size = batch_size # Batch size
        self.name = 'SDNN' # name of the model
        self.flag = 0

        self.epoch = 0
        self.best_acc = 0
        self.best_acc_prune = 0
        self.now_acc = 0
        self.connection_count = 0

        self.scheme = scheme


    # Forward pass
    # we first compute the hidden activations by utilizing the input to hidden weights (w1) , and hidden to hidden weights,
    # (w2) and their corresponding mask matrices (m1, m2), as well as the hidden biases (b1)
    # After computing the hidden activations, we use it alongside hidden to out weight matrix (w3) and corresponding
    # mask (m3), and input to output connections (w4) and its corresponding mask (m4), and output biases (b2)
    # at the end we return the computed output
    def forward(self, x, retain_grad = True):
        self.hidden = torch.zeros(x.size(0), self.max_size)

        for i,j in enumerate(self.active_index):
            self.hidden[:, j] = F.relu((torch.mm(self.hidden.clone(), torch.mul(self.w2[:, j], self.m2[:, j]).view(-1, 1))
                                       + torch.mm(x, torch.mul(self.w1[:, j], self.m1[:, j]).view(-1,1))
                                       + self.b1[:, j])).squeeze(1)

        out = torch.mm(self.hidden, torch.mul(self.w3, self.m3)) \
                  + torch.mm(x, torch.mul(self.w4, self.m4)) \
                  + self.b2

        if retain_grad:
            out.retain_grad()

        return out


    # used in connection growth
    def forwardMask(self, display=True):
        for i,j in enumerate(self.active_index):
            mask_idx = list(set(range(self.max_size)) - set(self.active_index[:i]))
            self.m2.data[:, j][mask_idx] = 0
        if display:
            print('Forward mask, m2: %d' %np.count_nonzero(self.m2.data))


    def backwardGrad(self, outgrad):
        self.hidden.grad = torch.mm(outgrad, torch.t(self.w3))
        rev_idx = np.flip(self.active_index, axis=0)
        for i,j in enumerate(rev_idx):
            for k in range(i):
                self.hidden.grad.data[:, j] = self.hidden.grad.data[:, j] + self.hidden.grad.data[:, k] \
                                                  *self.w2.data[j, k]


    def displayConnection(self, display=True):
        """it shows the number of active weights in m1, m2, m3, and m4 masks"""
        m1 = 0
        m2 = 0
        m3 = 0
        m4 = np.count_nonzero(self.m4.data)
        for i,j in enumerate(self.active_index):
            m1 += np.count_nonzero(self.m1.data[:, j])
            m3 += np.count_nonzero(self.m3.data[j, :])
            for k in range(i):
                m2 += np.count_nonzero(self.m2.data[self.active_index[k]][j])

        if display:
            print('Connection Info: ')
            print('m1: %d, m2: %d, m3: %d, m4: %d' %(m1,m2,m3,m4))
            print('Total: %d' % (m1+m2+m3+m4))
        return m1, m2, m3, m4, m1+m2+m3+m4


    def save_checkpoint(self, state, is_best, folder_to_save, filename = '_checkpoint.pth.tar'):
        name_to_save = os.path.join(folder_to_save, self.name + filename)
        torch.save(state, name_to_save)
        if is_best:
            shutil.copyfile(name_to_save, os.path.join(folder_to_save, self.name + '_model_best.pth.tar'))
            print(f"also saved as the best checkpoint to {os.path.join(folder_to_save, self.name + '_model_best.pth.tar')}")
