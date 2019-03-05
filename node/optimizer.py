from node.node import *
import numpy as np

class Optimizer(object):
    """
    オプティマイザーの基底クラス
    """

    def __init__(self, parameters):

        self.parameters = parameters

    def optimize(self):

        raise(NotImplementedError)

class SGD(Optimizer):
    """
    モーメンタム:
        Δw_{t} <- -1. * (lerning rate) * (δE/δw) + (momentum) * Δw_{t-1}
        w <- w + Δw_{t}
    """

    def __init__(self, parameters, learning_rate=0.01, decay=0., **kwargs):

        super().__init__(parameters)

        self.learning_rate = learning_rate

        # Wight Decayの程度をコントロールするハイパーパラメーター
        # 各重みのフロベニウスノルムの二乗の和に掛けられる
        self.decay = decay

    def __call__(self):

        for i in range(len(self.parameters)):
            grad = self.parameters[i].grad
            delta = - self.learning_rate * grad

            if self.decay != 0.:
                delta -= self.learning_rate * self.decay * self.parameters[i].value 

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
        self.m = [np.zeros(parameter.value.shape) for parameter in self.parameters]
        self.v = [np.zeros(parameter.value.shape) for parameter in self.parameters]

        self.iteration = 0

    def __call__(self):

        self.iteration += 1

        for i, parameter in enumerate(self.parameters):
            # パラメータの勾配の1次と2次のモーメントの移動平均を更新する。
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * parameter.grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (parameter.grad**2)

            # モーメントの移動平均を使って勾配のモーメントの期待値の近似値を計算する。
            m_hat = self.m[i] / (1. - self.beta_1 ** self.iteration)
            v_hat = self.v[i] / (1. - self.beta_2 ** self.iteration)

            delta = -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps) -self.learning_rate * self.decay * (parameter.value)
            self.parameters[i].update(delta)

    def zero_grad(self):

        for i in range(len(self.parameters)):
            self.parameters[i].zero_grad()
        