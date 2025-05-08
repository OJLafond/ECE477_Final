from model import SDNN
import numpy as np
import os

def run_training_scheme(X_train, y_train, X_test, y_test, scheme, save_dir):
    params = {
        'A': {'init_size': 20, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.3, 'loop_num': 1}, #10
        'B': {'init_size': 100, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.9, 'loop_num': 1}, #10
        'C': {'init_size': 100, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.9, 'loop_num': 1} #10
    }[scheme]

    model = SDNN(in_num=X_train.shape[1], out_num=2,
                 init_size=params['init_size'], max_size=params['max_size'],
                 batch_size=256, scheme=scheme)
    model.structureInit(sparse=params['sparse'], ratio=params['sparse_ratio'])
    model.loadData(X_train, y_train, X_test, y_test, mode='train')
    model.train(duration=1, folder_to_save=save_dir) #10

    for _ in range(params['loop_num']):
        model.addConnection()
        model.train(duration=1, folder_to_save=save_dir) #10
        if scheme == 'A':
            model.cellDivision()
        else:
            model.pruneConnections()
        model.train(duration=1, folder_to_save=save_dir) #10
