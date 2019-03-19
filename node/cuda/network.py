from ..network import Network 

import numpy as np 
import cupy as cp 

def gpu(self):
    """
    パラメーターのデータをGPUに移動する。
    """
    parameters = []
    for layer in self.layers: 
        for parameter in layer.get_parameters():
            parameter.value = cp.array(parameter.value)

    return self

setattr(Network, "gpu", gpu)