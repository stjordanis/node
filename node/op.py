try:
    import cupy as np
except:
    import numpy as np

import itertools

class Op(object):
    def __init__(self):
        self.cache = []

    def register(self, *cache):
        self.cache += cache



############################################
###  Add / Subtract / Multiply / Divide  ###
############################################



class Add(Op):

    def __init__(self, x, y, *args):
        super(Add, self).__init__()
        self.register(x, y)
        self.output = self.forward()

    def forward(self):
        x, y = self.cache
        return x.value + y.value

    def backward(self, error):
        x, y = self.cache
        x.accumulate(error)
        x.accumulate(error)


class Subtract(Op):

    def __init__(self, x, y, *args):
        super(Subtract, self).__init__()
        self.register(x, y)
        self.output = self.forward()

    def forward(self):
        x, y = self.cache
        return x.value - y.value

    def backward(self, error):
        x, y = self.cache
        x.accumulate(error)
        y.accumulate(-error)


class Multiply(Op):

    def __init__(self, x, y, *args):
        super(Multiply, self).__init__()
        self.register(x, y)
        self.output = self.forward()

    def forward(self):
        x, y = self.cache
        return x.value * y.value

    def backward(self, error):
        x, y = self.cache
        x.accumulate(error * y.value)
        y.accumulate(error * x.value)


class Divide(Op):

    def __init__(self, x, y, *args):
        super(Divide, self).__init__()
        self.register(x, y)
        self.output = self.forward()

    def forward(self):
        x, y = self.cache
        return x.value / y.value

    def backward(self, error):
        x, y = self.cache
        x.accumulate(error / y.value)
        y.accumulate(-error * x.value / (y.value ** 2))



###########################
###  Matrix Operations  ###
###########################



class Dot(Op):

    def __init__(self, x, y, *args):
        super(Dot, self).__init__()
        self.register(x, y)
        self.output = self.forward()

    def forward(self):
        x, y = self.cache
        return np.dot(x.value, y.value)

    def backward(self, error):
        x, y = self.cache
        x.accumulate(np.dot(error, y.value.T))
        y.accumulate(np.dot(x.value.T, error))


class T(Op):

    def __init__(self, x, *args):
        super(T, self).__init__()
        self.register(x)
        self.output = self.forward()

    def forward(self):
        x = self.cache[0]
        return x.value.T

    def backward(self, error):
        x = self.cache[0]
        x.accumulate(error.T)


class Transpose(Op):

    def __init__(self, x, *args):
        super(Transpose, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, permutation = self.cache

        # 逆置き換えを計算
        inv = [0] * len(permutation)
        for i, j in enumerate(permutation):
            inv[j] = i

        self.register(inv)
        return x.value.transpose(*permutation)

    def backward(self, error):
        x, _, inv = self.cache
        x.accumulate(error.transpose(*inv))


class Reshape(Op):

    def __init__(self, x, *args):
        super(Reshape, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, shape = self.cache
        return x.value.reshape(*shape)

    def backward(self, error):
        x, shape = self.cache
        x.accumulate(error.reshape(*x.value.shape))



####################
###  Mean / Sum  ###
####################



class Mean(Op):

    def __init__(self, x, *args):
        super(Mean, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, axis = self.cache
        return np.mean(x.value, axis=axis, keepdims=True)

    def backward(self, error):
        x, axis = self.cache
        shape = [1] * len(x.value.shape)
        shape[axis] = x.value.shape[axis]
        x.accumulate(error * np.tile(1, shape) / x.value.shape[axis])


class Sum(Op):

    def __init__(self, x, *args):
        super(Sum, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, axis = self.cache
        return np.sum(x.value, axis=axis, keepdims=True)

    def backward(self, error):
        x, axis = self.cache
        shape = [1] * len(x.value.shape)
        shape[axis] = x.value.shape[axis]
        x.accumulate(np.tile(error, shape))



################################
###  Pow / Exp / Log / Sqrt  ###
################################



class Pow(Op):

    def __init__(self, x, *args):
        super(Pow, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, y = self.cache
        return x.value ** y

    def backward(self, error):
        x, y = self.cache
        x.accumulate(error * y * (x.value ** (y - 1)))


class Exp(Op):

    def __init__(self, x, *args):
        super(Exp, self).__init__()
        self.register(x)
        self.output = self.forward()

    def forward(self):
        x = self.cache[0]
        return np.exp(x.value)

    def backward(self, error):
        x = self.cache[0]
        x.accumulate(error * self.output)


class Log(Op):

    def __init__(self, x, *args):
        super(Log, self).__init__()
        self.register(x)
        self.output = self.forward()

    def forward(self):
        x = self.cache[0]
        return np.log(x.value)

    def backward(self, error):
        x = self.cache[0]
        x.accumulate(error / x.value)


class Sqrt(Op):

    def __init__(self, x, *args):
        super(Sqrt, self).__init__()
        self.register(x)
        self.output = np.sqrt(x.value)

    def forward(self):
        x = self.cache[0]
        return np.sqrt(x.value)

    def backward(self, err_sig):
        x = self.cache[0]
        x.accumulate(err_sig / (2 * self.output))



################
###  Others  ###
################



class Repeat(Op):

    def __init__(self, x, *args):
        super(Repeat, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, axis, times, _ = self.cache
        return np.repeat(x.value, times, axis=axis)

    def backward(self, err_sig):
        x, axis, times, keepdims = self.cache
        x.accumulate(np.mean(err_sig, axis=axis, keepdims=keepdims))


class Expand(Op):

    def __init__(self, x, *args):
        super(Expand, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, axis = self.cache
        return np.expand_dims(x.value, axis=axis)

    def backward(self, error):
        x, axis = self.cache
        x.accumulate(np.squeeze(error, axis=axis))


class Max(Op):

    def __init__(self, x, *args):
        super(Max, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, axis = self.cache
        idx = np.argmax(x.value, axis=axis)
        self.register(idx)
        return np.max(x.value, axis=axis)

    def backward(self, error):
        x, axis, idx = self.cache
        dx = np.zeros(x.value.shape)
        dx[np.arange(idx.size), idx.flatten()] = error.flatten()
        x.accumulate(dx)



#####################
###  Activations  ###
#####################



class Sigmoid(Op):

    def __init__(self, x, *args):
        super(Sigmoid, self).__init__()
        self.register(x, 32.5387)
        self.output = self.forward()

    def forward(self):
        x, alpha = self.cache
        return 1 / (1 + np.exp(-np.clip(x.value, -alpha, alpha)))

    def backward(self, error):
        x, _ = self.cache
        x.accumulate(error * self.output * (1 - self.output))


class Tanh(Op):

    def __init__(self, x, *args):
        super(Tanh, self).__init__()
        self.register(x, 32.5387)
        self.output = self.forward()

    def forward(self):
        x, alpha = self.cache
        return np.tanh(np.clip(x.value, -alpha, alpha))

    def backward(self, error):
        x, alpha = self.cache
        z = np.clip(x.value, -alpha, alpha)
        x.accumulate(error * 4 / ((np.exp(z) + np.exp(-z)) ** 2))


class ReLU(Op):

    def __init__(self, x, *args):
        super(ReLU, self).__init__()
        self.register(x)
        self.output = self.forward()

    def forward(self):
        x = self.cache[0]
        return np.maximum(0, x.value)

    def backward(self, error):
        x = self.cache[0]
        x.accumulate(error * (1 * (x.value > 0)))


class LeakyReLU(Op):

    def __init__(self, x, *args):
        super(LeakyReLU, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, alpha = self.cache
        return np.maximum(x.value * alpha, x.value)

    def backward(self, error):
        x, alpha = self.cache
        x.accumulate(error * np.where(x.value > 0, 1, alpha))


class SeLU(Op):

    def __init__(self, x, *args):
        super(SeLU).__init__()
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        self.register(x, alpha, scale)
        self.output = self.forward()

    def forward(self):
        x, alpha, scale = self.cache
        return scale * np.where(x.value >= 0, x.value, alpha * np.exp(x.value)) - alpha

    def backward(self, error):
        x, alpha, scale = self.cache
        x.accumulate(error * scale * np.where(x.value >= 0, 1, alpha * np.exp(x.value)))



########################
###  Loss Functions  ###
########################



class BinaryCrossEntropy(Op):

    def __init__(self, x, y, *args):
        super(BinaryCrossEntropy, self).__init__()
        self.register(x, y, 1e-5)
        self.output = self.forward()

    def forward(self):
        x, y, alpha = self.cache
        x.value = np.clip(x.value, alpha, 1-alpha)
        z = -y.value * np.log(x.value) -(1 - y.value) * np.log(1 - x.value)
        return np.mean(z)

    def backward(self, error):
        x, y, _ = self.cache
        shape = [1] * len(x.value.shape)
        x.accumulate(np.tile(1, shape) / x.value.shape[0] \
                    * (x.value - y.value) / (x.value * (1 - x.value)))


class SoftmaxWithBinaryCrossEntropy(Op):

    def __init__(self, x, y, *args):
        super(SoftmaxWithBinaryCrossEntropy, self).__init__()
        self.register(x, y, 1e-5)
        self.output = self.forward()

    def forward(self):
        x, y, alpha = self.cache
        z = np.clip(self.softmax(x.value), alpha, 1-alpha)
        self.register(z)
        return self.binary_cross_entropy(z, y.value)

    def softmax(self, x, axis=1):
        z = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return z / np.sum(z, axis=axis, keepdims=True)

    def binary_cross_entropy(self, x, y):
        z = y * np.log(x) + (1 - y) * np.log(1 - x)
        return -1 * np.mean(np.sum(z, axis=1, keepdims=False), axis=0, keepdims=False)

    def backward(self, error):
        x, y, alpha, z = self.cache
        x.accumulate(z - y.value)


class MeanSquaredError(Op):

    def __init__(self, x, y, *args):
        super(MeanSquaredError, self).__init__()
        self.register(x, y)
        self.output = self.forward()

    def forward(self):
        x, y = self.cache
        return 0.5 * np.mean((x.value - y.value) ** 2, keepdims=False)

    def backward(self, error):
        x, y = self.cache
        x.accumulate(x.value - y.value)



######################
###  Convolutions  ###
######################



class Lower(Op): # => Im2Col

    def __init__(self, x, *args):
        super(Lower, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, kernel, stride, pad = self.cache
        N, C, H, _ = x.value.shape
        shape = (H + 2 * pad - kernel) // stride + 1
        y = np.pad(x.value, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
        z = np.zeros([N, C, kernel, kernel, shape, shape])

        # サイズkのカーネルが与えられるとする。
        # すると、z[..i,j..]には
        # 1) 各バッチ
        # 2) 各チャンネル
        # 3) 各畳み込み領域
        # においてカーネルのi行j列の値が保存される
        for i, j in itertools.product(range(kernel), repeat=2):
            z[:, :, i, j, :, :] = \
                y[:, :, i:(i+stride*shape):stride, j:(j+stride*shape):stride]

        return z.transpose(0, 4, 5, 1, 2, 3).reshape(N * (shape ** 2), -1)

    def backward(self, error):
        x, kernel, stride, pad = self.cache
        N, C, H, _ = x.value.shape
        shape = (H + 2 * pad - kernel) // stride + 1
        error = error.reshape(N, shape, shape, C, kernel, kernel)
        error = error.transpose(0, 3, 4, 5, 1, 2)
        dx = np.zeros([N, C, H + 2 * pad + stride - 1, H + 2 * pad + stride - 1])
        for i, j in itertools.product(range(kernel), repeat=2):
            dx[:, :, i:(i+stride*shape):stride, j:(j+stride*shape):stride] \
                += error[:, :, i, j, :, :]
        x.accumulate(dx[:, :, pad:H+pad, pad:H+pad])

class Higher(Op): # => Col2Im

    def __init__(self, x, *args):
        """
        引数
            x                   入力
            mini_batch_size     ミニバッチのサイズ
            output              出力のサイズ
            num_in_ch           入力のチャンネル数
            kernel              カーネルのサイズ
            stride              ストライドのサイズ
            pad                 パッドのサイズ
        """
        super(Higher, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, mini_batch_size, output, num_in_ch, kernel, stride, pad = self.cache

        # 参考
        # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
        a = (output + 2 * pad - kernel) % stride
        input = stride * (output - 1) + a + kernel - 2 * pad

        z = x.value.reshape(mini_batch_size,
                            output,
                            output,
                            num_in_ch,
                            kernel,
                            kernel)
        z = z.transpose(0, 3, 4, 5, 1, 2)

        y = np.zeros([mini_batch_size,
                      num_in_ch,
                      input + 2 * pad,
                      input + 2 * pad])

        for i, j in itertools.product(range(kernel), repeat=2):
            y[:, :, i:(i+stride*output):stride, j:(j+stride*output):stride] \
                += z[:, :, i, j, :, :]

        return y[:, :, pad:input+pad, pad:input+pad]

    def backward(self, error):
        x, mini_batch_size, output, num_in_ch, _, kernel, stride, pad = self.cache

        error = np.pad(error,
                       [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                       "constant")

        dx = np.zeros([mini_batch_size,
                       num_in_ch,
                       kernel,
                       kernel,
                       output,
                       output])

        for i, j in itertools.product(range(kernel), repeat=2):
            dx[:, :, i, j, :, :] = error[:, :, i:(i+stride*output):stride, j:(j+stride*output):stride]

        dx = dx.transpose(0, 4, 5, 1, 2, 3)
        dx = dx.reshape(mini_batch_size * (output ** 2), -1)
        x.accumulate(dx)



#############################
###  Batch Normalization  ###
#############################



class BatchNormalization(Op):
    # 参考
    # https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py

    def __init__(self, x, *args):
        """
        引数
            x: 入力
            gamma           正規化後の値をスケールするパラメーター
            beta            正規化後の値をシフトするパラメーター
            eps             ゼロ除算を防ぐための値
            is_train        Trueならrunning_meanとrunning_varの値を更新
            running_mean
            running_var
            alpha
        """
        super(BatchNormalization, self).__init__()
        self.register(x, *args)
        self.output = self.forward()

    def forward(self):
        x, gamma, beta, eps, is_train, running_mean, running_var, alpha = self.cache

        if is_train:
            mu = np.mean(x.value, axis=0)
            xc = x.value - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + eps)
            xn = xc / std
            self.register(xc, xn, std)

            running_mean.value = alpha * running_mean.value + (1 - alpha) * mu
            running_var.value = alpha * running_var.value + (1 - alpha) * var

        else:
            mu = running_mean.value
            std = np.sqrt(running_var.value + eps)

        return gamma.value * (x.value - mu) / std + beta.value

    def backward(self, error):
        x, gamma, beta, eps, _, _, _, _, xc, xn, std = self.cache

        beta.accumulate(np.mean(error, axis=0))
        gamma.accumulate(np.sum(xn * error, axis=0))

        dxn = gamma.value * error
        dxc = dxn / std
        dstd = -np.sum((dxn * xc) / (std * std), axis=0)
        dvar = 0.5 * dstd / std
        dxc += (2 / x.value.shape[0]) * xc * dvar
        dmu = np.sum(dxc , axis=0)

        x.accumulate(dxc - dmu / x.value.shape[0])
