import numpy as np
import gzip

def load_mnist_labels(path):
    with gzip.open(path) as f:
        raw_data = f.read() 
        label_num = int.from_bytes(raw_data[4:8], "big")
        labels = np.frombuffer(raw_data, np.uint8, offset=8).reshape(label_num, 1)

    return labels

def load_mnist_images(path, flatten=False):
    with gzip.open(path) as f:
        # Refer to https://weblabo.oscasierra.net/python/ai-mnist-data-detail.html for how MNIST files are constructed
        raw_data = f.read()
        image_num = int.from_bytes(raw_data[4:8], "big")
        row_size = int.from_bytes(raw_data[8:12], "big")
        col_size = int.from_bytes(raw_data[12:16], "big")
        # ! np.uint8 corresponds to unsigned byte
        images = np.frombuffer(raw_data, np.uint8, offset=16)
        if not flatten:
            images = images.reshape(image_num, 1, row_size, col_size)
        else:
            images = images.reshape(image_num, row_size*col_size)

    return images

class Dataset(object):

    def __len__(self):
        raise(NotImplementedError)

    def __getitem__(self, i):
        raise(NotImplementedError)

class MNIST(Dataset):

    def __init__(self, train=True, flatten=False):
        if train:
            paths = ["../datasets/mnist/train-images-idx3-ubyte.gz", 
                     "../datasets/mnist/train-labels-idx1-ubyte.gz"]
        else:
            paths = ["../datasets/mnist/t10k-images-idx3-ubyte.gz", 
                     "../datasets/mnist/t10k-labels-idx1-ubyte.gz"]

        self.inputs = load_mnist_images(paths[0], flatten)
        self.targets = load_mnist_labels(paths[1])

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, i):
        # ! 正解ラベルを1-hotベクトルに変換する。
        return (self.inputs[i],
                np.eye(10)[self.targets[i]].reshape(10))