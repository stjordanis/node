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
        x.acc_grad(error)
        x.acc_grad(error)


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
        x.acc_grad(error)
        y.acc_grad(-error)


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
        x.acc_grad(error * y.value)
        y.acc_grad(error * x.value)


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
        x.acc_grad(error / y.value)
        y.acc_grad(-error * x.value / (y.value ** 2))



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
        x.acc_grad(np.dot(error, y.value.T))
        y.acc_grad(np.dot(x.value.T, error))


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
        x.acc_grad(error.T)


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
        x.acc_grad(error.transpose(*inv))


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
        x.acc_grad(error.reshape(*x.value.shape))



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
        x.acc_grad(error * np.tile(1, shape) / x.value.shape[axis])


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
        x.acc_grad(np.tile(error, shape))



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
        x.acc_grad(error * y * (x.value ** (y - 1)))


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
        x.acc_grad(error * self.output)


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
        x.acc_grad(error / x.value)


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
        x.acc_grad(err_sig / (2 * self.output))



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
        x.acc_grad(np.mean(err_sig, axis=axis, keepdims=keepdims))


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
        x.acc_grad(np.squeeze(error, axis=axis))


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
        x.acc_grad(dx)



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
        x.acc_grad(error * self.output * (1 - self.output))


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
        x.acc_grad(error * 4 / ((np.exp(z) + np.exp(-z)) ** 2))


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
        x.acc_grad(error * (1 * (x.value > 0)))


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
        x.acc_grad(error * np.where(x.value > 0, 1, alpha))


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
        x.acc_grad(error * scale * np.where(x.value >= 0, 1, alpha * np.exp(x.value)))



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
        x.acc_grad(np.tile(1, shape) / x.value.shape[0] \
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
        x.acc_grad(z - y.value)


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
        x.acc_grad(x.value - y.value)



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
        x.acc_grad(dx[:, :, pad:H+pad, pad:H+pad])

class Higher(Op): # => Col2Im

    def __init__(self, x, mini_batch_size, output_size, num_in_ch, num_out_ch, filter_size, stride=1, pad=0):
        super(Higher, self).__init__()

        # アウトプットの形を計算
        # --- 参考 ---
        # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html 
        a = (output_size + 2 * pad - filter_size) % stride
        input_size = stride * (output_size - 1) + a + filter_size - 2 * pad

        # y[:, :, i, j, :, :]が各畳み込みのi行j列の要素を指すように変形
        y = x.value.reshape(mini_batch_size, output_size, output_size, num_in_ch, filter_size, filter_size)
        y = y.transpose(0, 3, 4, 5, 1, 2)

        # 出力を埋める
        self.output = np.zeros(
            [
                mini_batch_size, 
                num_in_ch, 
                input_size + 2 * pad, 
                input_size + 2 * pad
            ]
        )
        for i, j in it.product(range(filter_size), repeat=2):
            stops = [i + stride * output_size, j + stride * output_size]
            self.output[:, :, i:stops[0]:stride, j:stops[1]:stride] += y[:, :, i, j, :, :]

        self.output = self.output[:, :, pad:input_size+pad, pad:input_size+pad]

        # バックワード演算用に保存
        super(Higher, self).__init__(x, 
                                     mini_batch_size,
                                     input_size,
                                     output_size,
                                     num_in_ch,
                                     num_out_ch,
                                     filter_size,
                                     stride,
                                     pad)

    def backward(self, err_sig):
        x, mini_batch_size, input_size, output_size, num_in_ch, num_out_ch, filter_size, stride, pad = self._srcs

        err_sig = np.pad(err_sig, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

        # 出力の型を用意する
        dx = np.zeros([mini_batch_size, num_in_ch, filter_size, filter_size, output_size, output_size])

        # 通常のLoweringのように畳み込む部分を埋める
        for i, j in it.product(range(filter_size), repeat=2):
            stops = [i + stride * output_size, j + stride * output_size]
            dx[:, :, i, j, :, :] = err_sig[:, :, i:stops[0]:stride, j:stops[1]:stride]

        # チャンネル数分の畳み込む部分を行ベクトルに持つ行列に変換
        dx = dx.transpose(0, 4, 5, 1, 2, 3).reshape(mini_batch_size*(output_size**2), -1)
        x.acc_grad(dx)