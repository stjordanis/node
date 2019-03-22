import numpy as np 
import cupy as cp 

from ..optimizer import Adam 

def __init__(self, 
             parameters, 
             learning_rate = 1e-3,
             decay = 0., 
             beta_1 = 0.9, 
             beta_2 = 0.999, 
             eps = 1e-8):

    self.parameters = parameters
    self.learning_rate = learning_rate
    self.decay = decay
    self.beta_1 = beta_1 
    self.beta_2 = beta_2 
    self.eps = eps

    # パラメータの勾配の1次と2次のモーメントの移動平均を0で初期化する。
    self.m = [cp.zeros(parameter.value.shape) for parameter in self.parameters]
    self.v = [cp.zeros(parameter.value.shape) for parameter in self.parameters]

    self.iteration = 0

setattr(Adam, "__init__", __init__)