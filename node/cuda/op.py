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

##################################
###                            ###
### Vector (Matrix) Operations ###
###                            ###
##################################

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

##############
###        ###
### Others ###
###        ###
##############

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

######################
###                ###
### Loss Functions ###
###                ###
######################

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

####################
###              ###
### Convolutions ###
###              ###
####################

class Lower(Op):
    """
    畳み込み演算を行列積に変換する。
    """

    def __init__(self, x, filter_size, stride=1, pad=0):
        """
        引数
            x: 入力
            filter_size: フィルターの大きさ
            stride: 畳み込みの間隔
            pad: パディングの大きさ
        """
        # 入力の形を取り出す
        # 高さと幅は同じなので、一つだけ使う
        N, C, S, _ = x.value.shape

        # 出力の形を計算する
        output_size = (S + 2 * pad - filter_size) // stride + 1

        # 入力にパディングする
        y = cp.pad(x.value, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

        super(Lower, self).__init__(x, filter_size, stride, pad, output_size)

        # 出力の型を用意する
        self.output = cp.zeros([N, C, filter_size, filter_size, output_size, output_size])

        # i:stops[0 or 1]:strideで各畳み込みのi行j列の値のインデックスを指定している
        # self.output[:, :, i, j, :, :]に各畳み込みのi行j列の値の行列を代入している
        for i, j in it.product(range(filter_size), repeat=2):
            stops = [i + stride * output_size, j + stride * output_size]
            self.output[:, :, i, j, :, :] = y[:, :, i:stops[0]:stride, j:stops[1]:stride]

        self.output = self.output.transpose(0, 4, 5, 1, 2, 3).reshape(N*(output_size**2), -1)

    def backward(self, err_sig):
        N, C, S, _ = self._srcs[0].value.shape

        # 誤差信号を行列から元の形に戻す
        err_sig = err_sig.reshape(N, self._srcs[4], self._srcs[4], C, self._srcs[1], self._srcs[1])
        err_sig = err_sig.transpose(0, 3, 4, 5, 1, 2)

        # 誤差信号を足し合わせる
        dx = cp.zeros([N, C, S + 2 * self._srcs[3] + self._srcs[2] - 1, S + 2 * self._srcs[3] + self._srcs[2] - 1])
        for i, j in it.product(range(self._srcs[1]), repeat=2):
            stops = [i + self._srcs[2] * self._srcs[4], j + self._srcs[2] * self._srcs[4]]
            dx[:, :, i:stops[0]:self._srcs[2], j:stops[1]:self._srcs[2]] += err_sig[:, :, i, j, :, :]

        self._srcs[0].acc_grad(dx[:, :, self._srcs[3]:S+self._srcs[3], self._srcs[3]:S+self._srcs[3]])