try:
    import cupy as np
except:
    import numpy as np

import random
import node.node as node

def stack(mini_batch):
    inputs, targets = [], []
    for input, target in mini_batch:
        inputs.append(input)
        targets.append(target)
    return (node.Node(np.array(inputs), no_grad=True), 
            node.Node(np.array(targets), no_grad=True))

class DataLoader(object):

    def __init__(self, dataset, mini_batch_size, *args):
        self.dataset = dataset
        self.mini_batch_size = mini_batch_size
        self.idx = [i for i in range(len(self.dataset))]
        self.i = 0
        random.shuffle(self.idx)

    def __iter__(self, *args):
        return self

    def __len__(self):
        return len(self.dataset) // self.mini_batch_size

    def __next__(self):
        if self.i + self.mini_batch_size > len(self.dataset) or self.i == len(self.dataset):
            self.i = 0
            random.shuffle(self.idx)
            raise(StopIteration)

        mini_batch = []
        for _ in range(self.mini_batch_size):
            mini_batch.append(self.dataset[self.idx[self.i]])
            self.i += 1

        return stack(mini_batch)