import numpy as np
import cupy as cp

from ..node import Node, _destruct_tree, TREE

def __init__(self, value, no_grad=False):
    self.value = value
    self._no_grad = no_grad
    
    # no_gradがTrueの場合、このNodeオブジェクトのインスタンスは計算グラフの葉なので、勾配のプロパティを必要としない。
    if not no_grad:
        self.grad = cp.zeros(self.value.shape)

setattr(Node, "__init__", __init__)

"""
Nodeインスタンスが持つデータのGPUやCPUへの転送方法を実装する。
"""

# CPU -> GPU
def gpu(self):
    self.value = cp.array(self.value.astype(np.float32))
    return self

# GPU -> CPU
def cpu(self):
    self.value = cp.asnumpy(self.value)
    return self

setattr(Node, "gpu", gpu)
setattr(Node, "cpu", cpu)