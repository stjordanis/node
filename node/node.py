try:
    import cupy as np
    DEVICE = "gpu"
except:
    import numpy as np
    DEVICE = "cpu"

message = "Mode: {}".format(DEVICE.upper())
print(message)

import collections
import node.op as op



# Operation instance (Op in op.py) which has info to compute backward operation and node (
# Node in this file) instance which has the result of the operation are saved in GRAPH.
#
# Example -- In Feed Forward Computation
#
# In: c = a + b     Out: GRAPH = [Pair(Add, c)]
# In: f = d * e     Out: GRAPH = [Pair(Add, c), Pair(Mul, f)]
Pair = collections.namedtuple("Pair", ("op", "node"))
GRAPH = []

# A pair of Op and Node is added to GRAPH if True
CONSTRUCT_GRAPH = True



#########################
###  Graph Operation  ###
#########################



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
    # Wrapper for operations which take 2 inputs i.e. Add, Subtract and so on.
    def wrapper(x, y, *args):
        z = fn(x, y, *args)
        node = Node(z.output)
        _add_new_pair(z, node)
        return node
    return wrapper


def _single_oprand_op(fn):
    # When constant is passed, it is needed to be converted to node.
    def wrapper(x, *args):
        if type(x) != Node:
            x = _core_scaler2node(x)
        y = fn(x, *args)
        node = Node(y.output)
        _add_new_pair(y, node)
        return node
    return wrapper



########################
###  Node Operation  ###
########################



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
    # It comes with to numpy broadcast rule: add 1 to the lead
    # Example -- shape is (1, 32) and x shape is (32)
    # x shape becomes (1, 32)
    for axis in range(len(shape) - len(x.value.shape)):
        x = x.expand(0)

    # Another numpy rule: adjust each dimension to shape
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
        * Argument
            value  the value of node
            off    self.grad is initialized only if off = True
            name   the name of node
        """
        self.value = value.astype(np.float32)
        self.off = off
        self.name = name

        if not self.off:
            self.grad = np.zeros(value.shape)



    ############################################
    ###  Add / Subtract / Multiply / Divide  ###
    ############################################

    # NOTE
    # 左に__add__が定義されたインスタンスがあり、__add__で右のインスタンスの__radd__を呼ば
    # ないような実装の場合、エラーが発生するので、左にNodeインスタンス、右に別のインスタン
    # スを置くようにする(__add__が優先される)。



    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __add__(self, x, *args):
        return op.Add(self, x, *args)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __radd__(self, x, *args):
        return op.Add(x, self, *args)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __sub__(self, x, *args):
        return op.Subtract(self, x, *args)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rsub__(self, x, *args):
        return op.Subtract(x, self, *args)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __mul__(self, x, *args):
        return op.Multiply(self, x, *args)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rmul__(self, x, *args):
        return op.Multiply(x, self, *args)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __truediv__(self, x, *args):
        return op.Divide(self, x, *args)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rtruediv__(self, x, *args):
        return op.Divide(x, self, *args)



    ###########################
    ###  Matrix Operations  ###
    ###########################



    @_two_oprand_op
    def dot(self, x, *args):
        return op.Dot(self, x, *args)

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
        if DEVICE == "gpu":
            return np.asnumpy(self.value)
        else:
            return self.value



################
###  Others  ###
################



def concatenate(x, axis=0):
    y = op.Concatenate(x, axis)
    node = Node(y.output)
    _add_new_pair(y, node)
    return node
