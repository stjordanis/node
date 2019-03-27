try:
    import cupy as np
except:
    import numpy as np

import node.node as node

class Optimizer(object):

    def __init__(self, parameters):
        self.parameters = parameters

class SGD(Optimizer):

    def __init__(self, 
                 parameters, 
                 learning_rate=1e-3, 
                 decay=0):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.decay = decay

    def __call__(self):
        for i, parameter in enumerate(self.parameters):
            delta = -self.learning_rate * parameter.grad -self.learning_rate * self.decay * parameter.value 
            self.parameters[i].update(delta)

    def zero_grad(self):
        for i in range(len(self.parameters)):
            self.parameters[i].zero_grad()

class Adam(Optimizer):

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
        self.m = [np.zeros(parameter.value.shape, dtype=np.float32) for parameter in self.parameters]
        self.v = [np.zeros(parameter.value.shape, dtype=np.float32) for parameter in self.parameters]

        self.iteration = 0

    def __call__(self):
        self.iteration += 1

        for i, parameter in enumerate(self.parameters):
            # パラメータの勾配の1次と2次のモーメントの移動平均を更新する。
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * parameter.grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (parameter.grad ** 2)

            # モーメントの移動平均を使って勾配のモーメントの期待値の近似値を計算する。
            m_hat = self.m[i] / (1. - self.beta_1 ** self.iteration)
            v_hat = self.v[i] / (1. - self.beta_2 ** self.iteration)

            delta = -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps) -self.learning_rate * self.decay * (parameter.value)
            self.parameters[i].update(delta)

    def zero_grad(self):
        for i in range(len(self.parameters)):
            self.parameters[i].zero_grad()
        