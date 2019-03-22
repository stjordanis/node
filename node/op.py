import numpy as np
import itertools as it

class Op(object):
    """
    各演算は_srcsプロパティに保存した値を使ってバックワード演算を行う。
    """

    def __init__(self, *srcs):
        """
        引数
            srcs: オペランドやバックワード演算に必要な値のリスト
        """
        self._srcs = srcs

##########################################################
###                                                    ###
### Addition / Subtraction / Multiplication / Division ###
###                                                    ###
##########################################################

class Add(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.output = self._srcs[0].value + self._srcs[1].value

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig)
        self._srcs[1].acc_grad(err_sig)

class Sub(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.output = self._srcs[0].value - self._srcs[1].value

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig)
        self._srcs[1].acc_grad(-err_sig)

class Mul(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.output = self._srcs[0].value * self._srcs[1].value

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig * self._srcs[1].value)
        self._srcs[1].acc_grad(err_sig * self._srcs[0].value)

class Div(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.output = self._srcs[0].value / self._srcs[1].value

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig / self._srcs[1].value)
        self._srcs[1].acc_grad(-1. * err_sig * self._srcs[0].value * (self._srcs[1].value**(-2)))

##################################
###                            ###
### Vector (Matrix) Operations ###
###                            ###
##################################

class Dot(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.output = np.dot(self._srcs[0].value, self._srcs[1].value)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(np.dot(err_sig, self._srcs[1].value.T))
        self._srcs[1].acc_grad(np.dot(self._srcs[0].value.T, err_sig))

class T(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        self.output = self._srcs[0].value.T 

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig.T)

class Transpose(Op):
    def __init__(self, x, *args):
        super().__init__(x)

        # 逆置き換えを作る
        self._inversed_fn = [0 for _ in range(len(args[0]))]
        for i, j in enumerate(args[0]):
            self._inversed_fn[j] = i

        self.output = self._srcs[0].value.transpose(*args[0])

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig.transpose(*self._inversed_fn))

class Reshape(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        self._original_shape = self._srcs[0].value.shape 
        self.output = self._srcs[0].value.reshape(*args[0])

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig.reshape(*self._original_shape))

#######################
###                 ###
### Mean / Sumation ###
###                 ###
#######################

class Mean(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        self._axis = args[0]
        self._original_shape = self._srcs[0].value.shape
        self.output = np.mean(self._srcs[0].value, axis=self._axis, keepdims=True)

    def backward(self, err_sig):
        _shape = [1 for _ in range(len(self._original_shape))]
        _shape[self._axis] = self._original_shape[self._axis]
        self._srcs[0].acc_grad(err_sig * np.tile(1, _shape) / self._srcs[0].value.shape[self._axis])

class Sum(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        self._axis = args[0]
        self._original_shape = self._srcs[0].value.shape
        self.output = np.sum(self._srcs[0].value, axis=self._axis, keepdims=True)

    def backward(self, err_sig):
        _shape = [1 for _ in range(len(self._original_shape))]
        _shape[self._axis] = self._original_shape[self._axis]
        self._srcs[0].acc_grad(np.tile(err_sig, _shape))

######################################################
###                                                ###
### Power / Exponential / Logarithm / Ssquare Root ###
###                                                ###
######################################################

class Pow(Op):
    def __init__(self, x, y, *args):
        super().__init__(x)
        self._y = y
        self.output = self._srcs[0].value ** 2

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig * 2 * (self._srcs[0].value))

class Exp(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        self.output = np.exp(self._srcs[0].value)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig * self.output)

class Log(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        # !
        self._seed = np.clip(self._srcs[0].value, 1e-100, 1e+100)
        self.output = np.log(self._seed)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig / self._seed)

class Sqrt(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        # ! オーバーフローを防ぐ
        # self.output = np.sqrt(np.clip(self._srcs[0].value, None, 1e+13))
        self.output = np.sqrt(np.clip(self._srcs[0].value, 1e-13, 1e+13))

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig / (2. * self.output))

##############
###        ###
### Others ###
###        ###
##############

class GetItem(Op):
    def __init__(self, x, idx):
        super().__init__(x)
        self._idx = idx
        self.output = self._srcs[0].value[self._idx]

    def backward(self, err_sig):
        mat = np.zeros(self._srcs[0].value.shape)
        mat[self._idx] += err_sig
        self._srcs[0].acc_grad(mat)

class TakeAlongAxis(Op):
    def __init__(self, x, indeces, axis):
        super().__init__(x)
        self._indeces = indeces.reshape(-1, 1)
        self._axis = axis
        self.output = np.take_along_axis(self._srcs[0].value, self._indeces, self._axis)

    def backward(self, err_sig):
        mat = np.zeros(self._srcs[0].value.shape)
        np.put_along_axis(mat, self._indeces, err_sig, self._axis)
        self._srcs[0].acc_grad(mat)

class Cat(Op):
    """
    Nodeインスタンスのリストを受け取り、それらのインスタンスを繋げたインスタンスを返す。
    """

    def __init__(self, x, *args):
        """
        引数
            x: Nodeインスタンスのリスト
            axis: 何次元で繋げるか
        """

        super().__init__(x)

        self.axis = args[0]

        # 各ノードの値を取り出してリストに追加していく
        # TODO 最適化
        self.output = []
        for t in range(len(self._srcs[0])):
            self.output.append(self._srcs[0][t].value)

        # 与えられた次元で繋げる
        self.output = np.concatenate(self.output, axis=args[0])

    def backward(self, err_sig):
        """
        上層から送られてきたエラーシグナルを分割してリストの各Nodeインスタンスに送る。
        """

        for t in range(err_sig.shape[self.axis]):
            x = np.take(err_sig, t, axis=self.axis)
            y = np.expand_dims(x, self.axis)
            self._srcs[0][t].acc_grad(y)

class Rep(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        # ! argsは(axis, times, np.mean計算後に次元を保存するか)で構成されている。
        self._args = args
        self.output = np.tile(self._srcs[0].value, self._args[1]).astype(np.float32)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(np.mean(err_sig, axis=self._args[0], keepdims=self._args[2]))

class Clip(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        self._min_bound = args[0]
        self._max_bound = args[1]
        self.output = np.clip(self._srcs[0].value, self._min_bound, self._max_bound)

    def backward(self, err_sig):
        # ! `and`だとエラーが起こるので、`&`を使う。
        _cond = (self._min_bound < self.output) & (self.output < self._max_bound)
        self._srcs[0].acc_grad(err_sig * np.where(_cond, 1, 0))

class Expand(Op):
    """
    新たな次元を追加する。
    """

    def __init__(self, x, *args):
        super(Expand, self).__init__(x, *args)
        self.output = np.expand_dims(x.value, axis=args[0])

    def backward(self, err_sig):
        self._srcs[0].acc_grad(np.squeeze(err_sig, axis=self._srcs[1]))

class Max(Op):
    """
    入力が与えられたとき、指定された軸方向の最大値を返す。
    """

    def __init__(self, x, *args):
        """
        引数
            args[0]: どの軸方向に最大値をとるか
        """
        indeces = np.argmax(x.value, args[0])
        indeces = np.expand_dims(indeces, axis=args[0])
        self.output = np.take_along_axis(x.value, indeces, axis=args[0])

        # 最大値のインデックスはバックワード演算時に使う
        super(Max, self).__init__(x, indeces, *args)

    def backward(self, err_sig):
        dx = np.zeros(self._srcs[0].value.shape)
        np.put_along_axis(dx, self._srcs[1], err_sig, axis=self._srcs[2])
        self._srcs[0].acc_grad(dx)


############################
###                      ###
### Activation Functions ###
###                      ###
############################

class Sigmoid(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        # オーバーフローが発生しないように、適切な値にクリッピングする。
        alpha = 32.538776394910684
        self._srcs[0].value = np.clip(x.value, -alpha, alpha)
        self.output = 1. / (1. + np.exp(-self._srcs[0].value))

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig * self.output * (1. - self.output))

class Tanh(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        # オーバーフローが発生しないように、適切な値にクリッピングする。
        alpha = 32.538776394910684
        self._srcs[0].value = np.clip(x.value, -alpha, alpha)
        self.output = np.tanh(self._srcs[0].value)

    def backward(self, err_sig):
        denominator = (np.exp(self._srcs[0].value) + np.exp(-self._srcs[0].value)) ** 2
        self._srcs[0].acc_grad(err_sig * 4 / denominator)

class ReLU(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        self.output = np.maximum(self._srcs[0].value, 0)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig * (1. * (self._srcs[0].value > 0)))

class LeakyReLU(Op):
    def __init__(self, x, *args):
        super().__init__(x)
        self.alpha = args[0]
        self.output = np.maximum(self._srcs[0].value, self._srcs[0].value*self.alpha)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(err_sig * np.where(self._srcs[0].value > 0, 1, self.alpha))

class SeLU(Op):
    """
    SeLU：
    Self-Normalizing Neural Networksで使われる活性化関数。レイヤーの重みの平均が0、標準偏差が√(1./i)で、かつ入力の平均が0、分散が1の場合、この活性化関数はStableなAttracting Fixed Point（この場合は出力の平均が0、分散が1に近づいていく）を持つ。

    元論文：
    https://arxiv.org/abs/1706.02515
    """

    def __init__(self, x, *args):
        super().__init__(x)
        self._alpha = 1.6732632423543772848170429916717
        self._scale = 1.0507009873554804934193349852946
        self.output = self._scale * np.where(
            self._srcs[0].value >= 0., 
            self._srcs[0].value, # 上の条件を満たす場合、y=xになる
            self._alpha * np.exp(np.clip(self._srcs[0].value, None, 1e+2)) - self._alpha
        )

    def backward(self, err_sig):
        grad = self._scale * np.where(
            self._srcs[0].value >= 0., 1., 
            self._alpha * np.exp(np.clip(self._srcs[0].value, None, 1e+2))
        )
        self._srcs[0].acc_grad(err_sig * grad)

class Softmax(Op):
    def __init__(self, x):
        super().__init__(x)
        self.output = self._softmax(self._srcs[0].value)

    def _softmax(self, x):
        _exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return _exp / np.sum(_exp, axis=1, keepdims=True)

    def backward(self, err_sig):
        # ! 基本的に推論時に使用されるので、バックワード演算は行わない。
        raise(NotImplementedError)

######################
###                ###
### Loss Functions ###
###                ###
######################

class BinaryCrossEntropy(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.shape = self._srcs[0].value.shape
        self._srcs[0].value = np.clip(self._srcs[0].value, 1e-12, 1-1e-12)
        xv = self._srcs[0].value
        yv = self._srcs[1].value
        self.output = - yv * np.log(xv) - (1 - yv) * np.log(1 - xv)
        self.output = np.mean(self.output)
    
    def backward(self, err_sig):
        shape = [1 for _ in range(len(self.shape))]
        #shape[0] = self.shape[0]
        err_sig = np.tile(1, shape) / self._srcs[0].value.shape[0]

        xv = self._srcs[0].value
        yv = self._srcs[1].value
        self._srcs[0].acc_grad(err_sig * (xv - yv) / (xv * (1 - xv)))

class SoftmaxWithCrossEntropy(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self._pred = self._softmax(self._srcs[0].value)
        self._pred = np.clip(self._pred, 1e-12, 1-1e-12)
        self.output = self._cross_entropy(self._pred, self._srcs[1].value)
        
    def _softmax(self, x):
        _exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return _exp / np.sum(_exp, axis=1, keepdims=True)

    def _cross_entropy(self, x, y):
        # ! np.logに十分に小さな値を渡すと、ゼロ除算が起こるので、np.clipを間に挟む
        _eq = y * np.log(x) + \
                (1. - y) * np.log(1. - x)
        return -1. * np.mean(np.sum(_eq, axis=1, keepdims=False), axis=0, keepdims=False)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(self._pred - self._srcs[1].value)

class MeanSquaredError(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.output = 0.5 * np.mean((x.value-y.value)**2, keepdims=False)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(self._srcs[0].value-self._srcs[1].value)

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
        y = np.pad(x.value, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

        super(Lower, self).__init__(x, filter_size, stride, pad, output_size)

        # 出力の型を用意する
        self.output = np.zeros([N, C, filter_size, filter_size, output_size, output_size])

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
        dx = np.zeros([N, C, S + 2 * self._srcs[3] + self._srcs[2] - 1, S + 2 * self._srcs[3] + self._srcs[2] - 1])
        for i, j in it.product(range(self._srcs[1]), repeat=2):
            stops = [i + self._srcs[2] * self._srcs[4], j + self._srcs[2] * self._srcs[4]]
            dx[:, :, i:stops[0]:self._srcs[2], j:stops[1]:self._srcs[2]] += err_sig[:, :, i, j, :, :]

        self._srcs[0].acc_grad(dx[:, :, self._srcs[3]:S+self._srcs[3], self._srcs[3]:S+self._srcs[3]])

class Higher(Op):
    """
    Lowringと逆の操作を行う
    """

    def __init__(self, x, output_size, num_in_ch, filter_size, stride=1, pad=0):
        # Nはバッチサイズと出力のサイズの積
        # Sはフィルターのサイズと入力のチャンネル数の積
        N, S = x.value.shape

        # アウトプットの形を計算
        # --- 参考 ---
        # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html 
        a = (output_size + 2 * pad - filter_size) % stride
        input_size = stride * (output_size - 1) + a + filter_size - 2 * pad

        # y[:, :, i, j, :, :]が各畳み込みのi行j列の要素を指すように変形
        y = x.value.reshape(N, output_size, output_size, num_in_ch, filter_size, filter_size)
        y = y.transpose(0, 3, 4, 5, 1, 2)

        # 出力を埋める
        self.output = np.zeros(
            [
                N, 
                num_in_ch, 
                input_size + 2 * pad + stride - 1, 
                input_size + 2 * pad + stride - 1
            ]
        )
        for i, j in it.product(range(filter_size), repeat=2):
            stops = [i + stride * output_size, j + stride * output_size]
            self.output[:, :, i:stops[0]:stride, j:stops[1]:stride] += y[:, :, i, j, :, :]

        # バックワード演算用に保存
        super(Higher, self).__init__(x, 
                                     input_size,
                                     output_size,
                                     num_in_ch,
                                     filter_size,
                                     stride,
                                     pad)

    def backward(self, err_sig):
        x, input_size, output_size, num_in_ch, filter_size, stride, pad = self._srcs

        # 出力の型を用意する
        dx = np.zeros([x.value.shape[0], num_in_ch, filter_size, filter_size, output_size, output_size])

        # 通常のLoweringのように畳み込む部分を埋める
        for i, j in it.product(range(filter_size), repeat=2):
            stops = [i + stride * output_size, j + stride * output_size]
            dx[:, :, i, j, :, :] = err_sig[:, :, i:stops[0]:stride, j:stops[1]:stride]

        # チャンネル数分の畳み込む部分を行ベクトルに持つ行列に変換
        dx = dx.transpose(0, 4, 5, 1, 2, 3).reshape(x.value.shape[0]*(output_size**2), -1)
        x.acc_grad(dx)