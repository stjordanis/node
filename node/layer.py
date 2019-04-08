try:
    import cupy as np
except:
    import numpy as np

import node.node as node

class Layer(object):

    def __init__(self):
        self.is_train = True

    def train(self):
        self.is_train = True

    def test(self):
        self.is_train = False

    def get_parameters(self):
        return list(self.parameters.values())


#############
###  MLP  ###
#############


class Linear(Layer):

    def __init__(self, num_in_units, num_h_units):
        super(Linear, self).__init__()

        self.parameters = {
            "W": node.Node(np.random.randn(num_in_units, num_h_units).astype(np.float32), name="W"),
            "b": node.Node(np.zeros(num_h_units, dtype=np.float32), name="b")
        }

    def __repr__(self):
        return "Linear"

    def __call__(self, input):
        return input.dot(self.parameters["W"]) + self.parameters["b"]



####################
### Convolutions ###
####################



class Convolution2D(Layer):

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 kernel,
                 stride=1,
                 pad=0,
                 use_bias=True):
        """
        引数
            num_in_ch   入力チャンネル数
            num_out_ch  出力チャンネル数
            kernel      カーネルサイズ
            strid       ストライドサイズ
            pad         ゼロパディング数
            use_bias    biasを使うかどうか
        """
        super(Convolution2D, self).__init__()

        self.kernel     = kernel
        self.stride     = stride
        self.pad        = pad
        self.use_bias   = use_bias

        self.parameters = {"W": node.Node(np.random.randn(num_out_ch,
                                                          num_in_ch,
                                                          kernel,
                                                          kernel).astype(np.float32),
                                          name="W")}
        if use_bias:
            self.parameters["b"] = node.Node(np.zeros(num_out_ch, dtype=np.float32),
                                             name="b")

    def __repr__(self):
        return "Convolution2D"

    def __call__(self, input):
        B, _, _, _ = self.parameters["W"].value.shape

        hidden = input
        hidden = hidden.lower(self.kernel, self.stride, self.pad)

        # `use_bias`が指定されている場合、畳込みのあとにバイアスを足し合わせる
        hidden = hidden.dot(self.parameters["W"].reshape(B, -1).t())
        if self.use_bias:
            hidden = hidden + self.parameters["b"]

        B, C, H, W = input.value.shape
        output = 1 + (H + 2 * self.pad - self.kernel) // self.stride
        return hidden.reshape(B, output, output, -1).transpose(0, 3, 1, 2)


class TransposedConvolution2D(Layer):
    # 畳み込み層との関係性
    # input -->       Convolution       --> output
    # input <-- Transposed  Convolution <-- output

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 kernel,
                 stride=1,
                 pad=0):
        """
        引数
            num_in_ch   上で示される関係性においてinputのチャンネル数
            num_out_ch  上で示される関係性においてoutputのチャンネル数
            kernel      カーネルサイズ
            stride      ストライドサイズ
            pad         ゼロパディング数
        """
        super(TransposedConvolution2D, self).__init__()

        self.num_in_ch  = num_in_ch
        self.num_out_ch = num_out_ch
        self.kernel     = kernel
        self.stride     = stride
        self.pad        = pad

        self.parameters = {"W": node.Node(np.random.randn(num_in_ch,
                                                          num_out_ch,
                                                          kernel,
                                                          kernel).astype(np.float32),
                                          name="W")}

    def __repr__(self):
        return "TransposedConvolution2D"

    def __call__(self, input):
        B, _, H, _ = input.value.shape
        hidden = input
        hidden = hidden.transpose(0, 2, 3, 1).reshape(B * (H ** 2), -1)
        hidden = hidden.dot(self.parameters["W"].reshape(self.num_in_ch, -1))
        hidden = hidden.higher(B, H, self.num_out_ch, self.kernel, self.stride, self.pad)
        return hidden


class MaxPooling2D(Layer):

    def __init__(self, kernel, stride=1, pad=0):
        super(MaxPooling2D, self).__init__()

        self.kernel = kernel
        self.stride = stride
        self.pad    = pad

        self.parameters = {}

    def __call__(self, input):
        hidden = input
        hidden = hidden.lower(self.kernel, self.stride, self.pad)
        hidden = hidden.reshape(-1, self.kernel ** 2)
        hidden = hidden.max(1)

        # 行列を元の形に戻す
        B, C, H, _ = input.value.shape
        output = 1 + (H - self.kernel) // self.stride
        return hidden.reshape(B, output, output, C).transpose(0, 3, 1, 2)



########################
###  Normalizations  ###
########################



class BatchNormalization(Layer):

    def __init__(self, num_in_units, alpha=0.9, eps=1e-5):
        """
        引数
            num_in_units   ユニット数(入力が4Dの時はチャンネル数)
            alpha          移動平均の更新率をコントロール
            eps            ゼロ除算を防ぐ
        """
        super(BatchNormalization, self).__init__()

        self.parameters = {
            "W": node.Node(np.random.randn(num_in_units).astype(np.float32), name="W"),
            "b": node.Node(np.zeros(num_in_units, dtype=np.float32), name="b")
        }

        self.alpha = alpha
        self.eps = eps

        self.running_mu = node.Node(np.zeros(num_in_units, dtype=np.float32))
        self.running_var = node.Node(np.ones(num_in_units, dtype=np.float32))

    def __repr__(self):
        return "BatchNormalization"

    def __call__(self, input):
        hidden = input

        if input.value.ndim != 2:
            N, C, H, W = input.value.shape
            hidden = hidden.transpose(0, 2, 3, 1)
            hidden = hidden.reshape(-1, input.value.shape[1])

        hidden = hidden.batch_normalization(self.parameters["W"],
                                            self.parameters["b"],
                                            self.eps,
                                            self.is_train,
                                            self.running_mu,
                                            self.running_var,
                                            self.alpha)

        if input.value.ndim != 2:
        	hidden = hidden.reshape(N, H, W, C)
        	hidden = hidden.transpose(0, 3, 1, 2)

        return hidden


class GroupNormalization(Layer):
    """
    Used when mini-batch size is not sufficient to compute statistics

    * Reference
    https://arxiv.org/abs/1803.08494
    """
    def __repr__(self):
        return "GroupNormalization"
