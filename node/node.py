try:
    import cupy as np
    DEVICE = "gpu"
except:
    import numpy as np
    DEVICE = "cpu"

message = "Works on {}".format(DEVICE.upper())
print(message)

import collections
import node.op as op

Pair = collections.namedtuple("Pair", ("op", "node"))
GRAPH = []

# Trueの時のみ計算グラフを構築
# 補足
# zero_gradでコントロール
CONSTRUCT_GRAPH = True



####################
###  Graphの操作  ###
####################



class zero_grad(object):

    def __enter__(self, *args):
        global CONSTRUCT_GRAPH
        CONSTRUCT_GRAPH = False

    def __exit__(self, *args):
        global CONSTRUCT_GRAPH
        CONSTRUCT_GRAPH = True


def _add_new_pair(op, node):
    global GRAPH

    if not CONSTRUCT_GRAPH:
        return

    node = Pair(op, node)
    GRAPH.append(node)


def _destruct_graph():
    global GRAPH
    GRAPH = GRAPH.__new__(list)


def _two_oprand_op(fn):
    def wrapper(x, y):
        op = fn(x, y)
        node = Node(op.output)
        _add_new_pair(op, node)
        return node
    return wrapper


def _single_oprand_op(fn):
    def wrapper(x, *args):
        if type(x) != Node:
            x = _core_scaler2node(x)
        op = fn(x, *args)
        node = Node(op.output)
        _add_new_pair(op, node)
        return node
    return wrapper



###################
###  Nodeの変形  ###
###################



def _core_scaler2node(x):
    return Node(np.array(x), off=True)


def _scaler2node(fn):
    def wrapper(x, y):
        if type(x) != Node:
            x = _core_scaler2node(x)
        elif type(y) != Node:
            y = _core_scaler2node(y)
        return fn(x, y)
    return wrapper


def _core_broadcast(x, shape):
    # 次元数が異なる場合、少ない次元を持つ方の先頭に足りない分だけ1を追加
    for axis in range(len(shape) - len(x.value.shape)):
        x = x.expand(0)

    # 各次元を最大値に合わせる
    for axis in range(len(shape)):
        if x.value.shape[axis] != shape[axis]:
            x = x.repeat(axis, shape[axis])

    return x


def _broadcast(fn):
    def wrapper(x, y):
        if x.value.shape != y.value.shape:
            shape = np.broadcast(x.value, y.value).shape
            x = _core_broadcast(x, shape)
            y = _core_broadcast(y, shape)
        return fn(x, y)
    return wrapper


class Node(object):

    def __init__(self, value, off=False, name=""):
        """
        引数
            value  この変数が持つ値を表す
            off    この変数が勾配を持つかを表す
            name   このノードの名前
        """
        self.value = value.astype(np.float32)
        self.off = off
        self.name = name

        # `off`が真のときのみ勾配を初期化
        if not self.off:
            self.grad = np.zeros(value.shape)



    ############################################
    ###  Add / Subtract / Multiply / Divide  ###
    ############################################



    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __add__(self, x):
        return op.Add(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __radd__(self, x):
        return op.Add(x, self)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __sub__(self, x):
        return op.Subtract(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rsub__(self, x):
        return op.Subtract(x, self)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __mul__(self, x):
        return op.Multiply(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rmul__(self, x):
        return op.Multiply(x, self)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __truediv__(self, x):
        return op.Divide(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rtruediv__(self, x):
        return op.Divide(x, self)



    ###########################
    ###  Matrix Operations  ###
    ###########################



    @_two_oprand_op
    def dot(self, x):
        return op.Dot(self, x)

    @_single_oprand_op
    def t(self):
        return op.T(self)

    @_single_oprand_op
    def transpose(self, *perm):
        return op.Transpose(self, perm)

    @_single_oprand_op
    def reshape(self, *shape):
        return op.Reshape(self, shape)



    ####################
    ###  Mean / Sum  ###
    ####################



    @_single_oprand_op
    def mean(self, axis=0):
        return op.Mean(self, axis)

    @_single_oprand_op
    def sum(self, axis=0):
        return op.Sum(self, axis)



    ####################
    ###  Mean / Sum  ###
    ####################



    @_single_oprand_op
    def __pow__(self, x):
        return op.Pow(self, x)

    @_single_oprand_op
    def exp(self):
        return op.Exp(self)

    @_single_oprand_op
    def log(self):
        return op.Log(self)

    @_single_oprand_op
    def sqrt(self):
        return op.Sqrt(self)



    ################
    ###  Others  ###
    ################



    @_single_oprand_op
    def repeat(self, axis, times, keepdims=True):
        return op.Repeat(self, axis, times, keepdims)

    @_single_oprand_op
    def expand(self, axis=1):
        return op.Expand(self, axis)

    @_single_oprand_op
    def max(self, axis=1):
        return op.Max(self, axis)



    #####################
    ###  Activations  ###
    #####################



    @_single_oprand_op
    def sigmoid(self):
        return op.Sigmoid(self)

    @_single_oprand_op
    def tanh(self):
        return op.Tanh(self)

    @_single_oprand_op
    def relu(self):
        return op.ReLU(self)

    @_single_oprand_op
    def leaky_relu(self, alpha=0.2):
        return op.LeakyReLU(self, alpha)

    @_single_oprand_op
    def selu(self):
        return op.SeLU(self)



    ########################
    ###  Loss Functions  ###
    ########################



    @_two_oprand_op
    def binary_cross_entropy(self, x):
        return op.BinaryCrossEntropy(self, x)

    @_two_oprand_op
    def softmax_with_binary_cross_entropy(self, x):
        return op.SoftmaxWithBinaryCrossEntropy(self, x)

    @_two_oprand_op
    def mean_squared_error(self, x):
        return op.MeanSquaredError(self, x)



    ######################
    ###  Convolutions  ###
    ######################



    @_single_oprand_op
    def lower(self, filter_size, stride=1, pad=0):
        return op.Lower(self, filter_size, stride, pad)

    @_single_oprand_op
    def higher(self,
               mini_batch_size,
               output,
               num_in_ch,
               kernel,
               stride=1,
               pad=0):
        return op.Higher(self,
                         mini_batch_size,
                         output,
                         num_in_ch,
                         kernel,
                         stride,
                         pad)



    #############################
    ###  Batch Normalization  ###
    #############################


    @_single_oprand_op
    def batch_normalization(self,
                            gamma,
                            beta,
                            eps,
                            is_train,
                            running_mu,
                            running_var,
                            alpha):
        return op.BatchNormalization(self,
                                     gamma,
                                     beta,
                                     eps,
                                     is_train,
                                     running_mu,
                                     running_var,
                                     alpha)

    def accumulate(self, grad):
        if not self.off:
            self.grad += grad

    def update(self, delta):
        self.value += delta

    def clear(self):
        self.grad.fill(0)

    def backward(self, error=None):
        for i, pair in enumerate(GRAPH[::-1]):
            if i == 0:
                if error is not None:
                    pair.op.backward(error)
                else:
                    pair.op.backward(np.ones(self.value.shape))
            else:
                pair.op.backward(pair.node.grad)
        _destruct_graph()

    def numpy(self):
        return np.asnumpy(self.value)
