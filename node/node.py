from . import op

import numpy as np
from collections import *

"""
動的計算グラフを表すリスト。各要素はあるOpとNodeのペアで構成され、バックワード計算の際に後ろから順に誤差信号を伝播する。
"""
Pair = namedtuple("Pair", ("op", "node"))
TREE = []

# このフラグがTrueの時だけ計算ブラフを構築する。
CONSTRUCT_TREE = True
class zero_grad(object):
    """
    主に類推時に使われる。定義されるwith構文の中では計算グラフが構築されない。
    """

    def __enter__(self, *args):

        global CONSTRUCT_TREE
        CONSTRUCT_TREE = False

    def __exit__(self, *args):

        global CONSTRUCT_TREE
        CONSTRUCT_TREE = True

def _add_new_pair(op, node):
    """
    計算グラフにあらたなペアを追加する。

    引数
        op: Opインスタンス
        node: Nodeインスタンス
    """

    # 計算グラフを構築しないなら、すぐNoneを返す。
    if not CONSTRUCT_TREE:
        return

    global TREE 
    node = Pair(op, node)
    TREE.append(node)

def _destruct_tree():
    """
    計算グラフをリセットする。
    """

    global TREE 
    TREE = TREE.__new__(list)

def _core_scaler2node(x):
    """
    実数値をNodeインスタンスに変換する。

    引数
        x: 数値型
    """

    # パラメーターではなく、葉であるので勾配を計算する必要がない。
    return Node(np.array(x), no_grad=True)

def _scaler2node(fn):
    """
    引数を調べてどちらかが数値型なら、Nodeインスタンスにに変換してからfnで定義される演算を行う。

    引数
        fn: Opクラス
    """

    def _wrapper(x, y):

        # どちらかが数値型の場合、Nodeインスタンスに変換してからfnで定義される演算を行う。
        if type(x) != Node:
            x = _core_scaler2node(x)

        elif type(y) != Node:
            y = _core_scaler2node(y)

        return fn(x, y)

    return _wrapper

def _core_broadcast(x, shape):
    """
    引数
        x: オペランド
        shape: ブロードキャスト後のオペランドの形 
    """
    # 次元数が足りなければ、先頭に次元を追加する
    for axis in range(len(shape) - len(x.value.shape)): 
        x = x.expand(0)

    for axis in range(len(shape)):
        if x.value.shape[axis] != shape[axis]:
            x = x.rep(axis, shape[axis])

    return x

def _broadcast(fn):
    """
    入力の形が異なる場合、一方に合わせて他方の形を変形する。
    """

    def _wrapper(x, y):

        # xとyをスワップすると、-など可換でないオペランドの場合にバグが発生するので、xとyを別々に考える必要がある。
        if x.value.shape != y.value.shape:
            shape = np.broadcast(x.value, y.value).shape
            x = _core_broadcast(x, shape)
            y = _core_broadcast(y, shape)

        return fn(x, y)

    return _wrapper

def _two_oprand_op(fn):
    """
    2つの値を取る演算子のデコレーター：
    2つの値を受け取り、fnで定義される演算子を適用する。値がスカラーである場合（Nodeオブジェクトのインスタンスではない）、それをNodeオブジェクトのインスタンスに変換したのち、必要な場合にブロードキャストすることで値間の次元を合わせる。次元と形が等しい2つのNodeオブジェクトのインスタンスが用意できたので、fnで定義される演算子を適用し、その出力で定義されるNodeオブジェクトのインスタンスとのペアを動的計算グラフに追加する。
    """

    def wrapper(x, y):
        _op = fn(x, y)
        _node = Node(_op.output)
        _add_new_pair(_op, _node)
        return _node

    return wrapper

def _single_oprand_op(fn):
    """
    1つの値を取る演算子のデコレーター：
    上の関数の値が1つの場合。
    """

    def wrapper(x, *args):
        # もしxがNodeオブジェクトのインスタンスではない場合、変換しておく。
        if type(x) != Node:
            x = _core_scaler2node(x)
        _op = fn(x, *args)
        _node = Node(_op.output)
        _add_new_pair(_op, _node)
        return _node

    return wrapper

def cat(x, axis=0):
    """
    Nodeインスタンスのリストを繋げる
    """

    op = op.Cat(x, axis)
    node = Node(op.output)
    others._add_new_pair(op, node)

    return node

class Node(object):
    def __init__(self, value, no_grad=False):
        self.value = value
        self._no_grad = no_grad
        
        # no_gradがTrueの場合、このNodeオブジェクトのインスタンスは計算グラフの葉なので、勾配のプロパティを必要としない。
        if not self._no_grad:
            self.grad = np.zeros(self.value.shape)

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
        return op.Sub(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rsub__(self, x):
        return op.Sub(x, self)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __mul__(self, x):
        return op.Mul(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rmul__(self, x):
        return op.Mul(x, self)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __truediv__(self, x):
        return op.Div(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rtruediv__(self, x):
        return op.Div(x, self)

    @_two_oprand_op
    def dot(self, x):
        return op.Dot(self, x)

    @_single_oprand_op
    def mean(self, axis=0):
        return op.Mean(self, axis)

    @_single_oprand_op
    def sum(self, axis=1):
        return op.Sum(self, axis)

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

    @_single_oprand_op
    def __getitem__(self, idx):
        return op.GetItem(self, idx)

    @_single_oprand_op
    def gather(self, indeces, axis=1):
        return op.TakeAlongAxis(self, indeces, axis)

    @_single_oprand_op
    def rep(self, axis=0, times=1, keepdims=True):
        return op.Rep(self, axis, times, keepdims)

    @_single_oprand_op
    def clip(self, min_bound=0., max_bound=np.inf):
        return op.Clip(self, min_bound, max_bound)

    @_single_oprand_op
    def expand(self, axis=1):
        return op.Expand(self, axis)

    @_single_oprand_op
    def max(self, axis=1):
        return op.Max(self, axis)

    @_single_oprand_op
    def t(self):
        return op.T(self)

    @_single_oprand_op
    def transpose(self, *perm):
        return op.Transpose(self, perm)

    @_single_oprand_op
    def reshape(self, *shape):
        return op.Reshape(self, shape)

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

    @_single_oprand_op
    def softmax(self):
        return op.Softmax(self)

    @_two_oprand_op
    def binary_cross_entropy(self, x):
        return op.BinaryCrossEntropy(self, x)

    @_two_oprand_op
    def softmax_with_cross_entropy(self, x):
        return op.SoftmaxWithCrossEntropy(self, x)

    @_two_oprand_op
    def mean_squared_error(self, x):
        return op.MeanSquaredError(self, x)

    @_single_oprand_op
    def lower(self, filter_size, stride=1, pad=0):
        return op.Lower(self, filter_size, stride, pad)

    @_single_oprand_op
    def higher(self, input_size, num_in_ch, filter_size, stride=1, pad=0):
        return op.Higher(self, input_size, num_in_ch, filter_size, stride, pad)

    def acc_grad(self, grad):
        if not self._no_grad:
            self.grad += grad

    def update(self, delta):
        self.value += delta

    def zero_grad(self):
        self.grad.fill(0.)

    def clear_tree(self):
        others._destruct_tree()

    def get_err_sig(self, shape):
        return np.ones(shape)

    def backward(self, err_sig=None):
        for i, pair in enumerate(TREE[::-1]):
            if i == 0:
                if err_sig is not None:
                    pair.op.backward(err_sig)
                else:
                    pair.op.backward(self.get_err_sig(self.value.shape))
            else:
                pair.op.backward(pair.node.grad)

        _destruct_tree()