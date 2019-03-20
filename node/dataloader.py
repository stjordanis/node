import numpy as np 
import random

from node.node import *

"""
Take an array list of input and target pair and stack them into a pair of a mini batch.
"""
def _stack(mini_batch):

    _pattern = []
    _target = []
    for _p, _t in mini_batch:
        _pattern.append(_p)
        _target.append(_t)
    
    return Node(np.array(_pattern)), Node(np.array(_target))

class DataLoader(object):

    def __init__(self, dataset, batch_size, *args):
        self._dataset = dataset
        self._batch_size = batch_size
        self._idx = [i for i in range(len(self._dataset))]
        random.shuffle(self._idx)
        self._i = 0

    def __iter__(self, *args):

        return self

    def __len__(self):

        return len(self._dataset) // self._batch_size

    def __next__(self):
        
        if self._i + self._batch_size > len(self._dataset) or self._i == len(self._dataset):
            self._i = 0
            random.shuffle(self._idx)
            raise(StopIteration)

        # get a mini-batch
        _mini_batch = []
        for _ in range(self._batch_size):
            _mini_batch.append(self._dataset[self._idx[self._i]])
            self._i += 1

        # convert the mini-batch into node
        return _stack(_mini_batch)