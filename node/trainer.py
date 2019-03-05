from node.node import *
from tqdm import tqdm
import numpy as np
import math

class Trainer(object):
    def __init__(self, net, optimizer, train_iter, val_iter):
        self._net = net 
        self._optimizer = optimizer
        self._train_iter = train_iter
        self._val_iter = val_iter 

    def train(self):
        _total = 0.
        _count = math.ceil(len(self._train_iter._dataset)/self._train_iter._batch_size)
        _prog_bar = tqdm(total=_count)
        for targets, inputs in self._train_iter:
            _prog_bar.update(1)
            outputs = self._net(inputs)
            loss = outputs.softmax_with_cross_entropy(targets)
            loss.backward()
            self._optimizer.step()
            _total += loss.v
        _prog_bar.close()
        print("Training Loss: {}".format(_total/_count))

    def evaluate(self):
        _total = 0.
        _count = math.ceil(len(self._val_iter._dataset)/self._val_iter._batch_size)
        _prog_bar = tqdm(total=_count)
        for targets, inputs in self._val_iter:
            _prog_bar.update(1)
            outputs = self._net(inputs)
            loss = outputs.softmax_with_cross_entropy(targets)
            _total += loss.v
            loss.clear_tree()
        _prog_bar.close()
        print("Validation Loss: {}".format(_total/_count))