from ..node import Node, Op
from ..node import _scaler2node, _broadcast
from ..node import _single_oprand_op, _two_oprand_op
from ..network import Network

import numpy as np 
import cupy as cp

"""
各演算をcuda対応版の演算にオーバーロードする。

使用ライブラリ:
    pycuda
    skcuda

対応する演算リスト:
    addition
"""

#######
# dot #
#######

class Dot(Op):

    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.output = cp.dot(self._srcs[0].value, self._srcs[1].value)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(cp.dot(err_sig, self._srcs[1].value.T))
        self._srcs[1].acc_grad(cp.dot(self._srcs[0].value.T, err_sig))

@_two_oprand_op
def dot(self, x):
    return Dot(self, x)

setattr(Node, "dot", dot)

################
# その他の演算 #
################

class Rep(Op):
    """
    Example:
    
    args[0] = 0
    args[1] = 3

    [1,2,3] 
    
    --->

    [[1,2,3],
     [1,2,3],
     [1,2,3]]
    """

    def __init__(self, x, *args):
        """
        引数
            args[0](int): どの軸方向に展開するか
            args[1](int): 何回展開するか
            args[2](bool): 計算後に次元を保存するか
        """
        super(Rep, self).__init__(x)

        self.axis = args[0]
        self.times = args[1]
        self.keepdims = args[2]

        self.output = cp.tile(x.value, self.times)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(cp.mean(err_sig, axis=self.axis, keepdims=self.keepdims))

@_single_oprand_op
def rep(self, axis=0, times=1, keepdims=True):
    return Rep(self, axis, times, keepdims)

setattr(Node, "rep", rep)

class SoftmaxWithCrossEntropy(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.mid = cp.clip(self.softmax(self._srcs[0].value), 1e-12, 1-1e-12)
        self.output = self.cross_entropy(self.mid, y.value)
        
    def softmax(self, x):
        exp = cp.exp(x - cp.max(x, axis=1, keepdims=True))
        return exp / cp.sum(exp, axis=1, keepdims=True)

    def cross_entropy(self, x, y):
        # ! np.logに十分に小さな値を渡すと、ゼロ除算が起こるので、np.clipを間に挟む
        eq = y * cp.log(x) + (1. - y) * cp.log(1. - x)
        return -1. * cp.mean(cp.sum(eq, axis=1, keepdims=False), axis=0, keepdims=False)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(self.mid - self._srcs[1].value)
        
@_two_oprand_op
def softmax_with_cross_entropy(self, x):
    return SoftmaxWithCrossEntropy(self, x)

setattr(Node, "softmax_with_cross_entropy", softmax_with_cross_entropy)