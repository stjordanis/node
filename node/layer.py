from . import node
import numpy as np 

class Layer(object):

    def __init__(self):

        # このフラグが立っている時に上の移動平均を更新する
        self.is_train = True 

    def get_parameters(self):

        raise(NotImplementedError)

    def get_parameters(self):
        
        return list(self.parameters.values())

class Linear(Layer):
    """
    Fully-Connectedレイヤー

    前のレイヤーのユニット数がmで隠れ層のユニット数がnだとすると、Wx + bを計算する(Wはn×m行列でbはエントリー数mのベクトル)
    """

    def __init__(self, num_in_units, num_h_units):

        super(Linear, self).__init__()

        # 前のレイヤーのニューロンとの結合につけられた重みと各ニューロンのバイアスを定義する
        self.parameters = {
            "W": node.Node(np.random.randn(num_in_units, num_h_units) * np.sqrt(1. / num_in_units)),
            "b": node.Node(np.zeros(num_h_units))
        }

    def __call__(self, x):

        return x.dot(self.parameters["W"]) + self.parameters["b"]

class BatchNorm1D(Layer):
    """
    バッチ正則化レイヤー

    参考
    https://arxiv.org/abs/1502.03167 (オリジナル)
    https://arxiv.org/abs/1702.03275
    """

    def __init__(self, num_in_units, alpha=0.9, eps=1e-8):
        """
        num_in_units := 前のレイヤーのユニット数
        alpha := 移動平均の更新率をコントロールするハイパーパラメーター
        """

        super(BatchNorm1D, self).__init__()

        self.parameters = {
            "W": node.Node(np.random.randn(num_in_units)),
            "b": node.Node(np.zeros(num_in_units))
        }

        self.alpha = alpha

        # 標準偏差で除算する際にゼロ除算エラーが発生しないようにする
        self.eps = eps

        # データセット全体の統計量をバッチごとに移動平均で計算された値で近似する
        # この値は推定時に使われる
        self.mu = np.zeros(shape=[1, num_in_units])
        self.sigma = np.ones(shape=[1, num_in_units])

    def __call__(self, x):
        
        # 訓練時はミニバッチの統計量を使って入力を正規化する
        if self.is_train:
            _mu = x.mean()
            _sigma = (((x - _mu) ** 2).mean() + self.eps).sqrt()

            # 全体の統計量を更新する
            self.mu = self.alpha * self.mu + (1 - self.alpha) * _mu.value
            self.sigma = self.alpha * self.sigma + (1 - self.alpha) * _sigma.value

        # 推定時は移動平均の値を使って入力を正規化する
        else:
            _mu = self.mu 
            _sigma = self.sigma

        return self.parameters["W"] * ((x - _mu) / _sigma) + self.parameters["b"]

class BatchNorm2D(Layer):
    
    def __init__(self, num_in_ch, alpha=0.9, eps=1e-8):
        """
        引数
            num_in_ch: 前のレイヤーのチャンネル数
            alpha: 移動平均の更新率をコントロールするハイパーパラメーター
            eps: ゼロ除算を防ぐ小さな値
        """

        super(BatchNorm2D, self).__init__()

        self.parameters = {
            "W": node.Node(np.random.randn(1, num_in_ch, 1, 1)),
            "b": node.Node(np.zeros([1, num_in_ch, 1, 1]))
        }

        self.alpha = alpha

        # 標準偏差で除算する際にゼロ除算エラーが発生しないようにする
        self.eps = eps

        # データセット全体の統計量をバッチごとに移動平均で計算された値で近似する
        # この値は推定時に使われる
        self.running_mu = np.zeros(shape=[1, num_in_ch, 1, 1])
        self.running_sigma = np.ones(shape=[1, num_in_ch, 1, 1])

    def __call__(self, x):
        
        # 訓練時はミニバッチの統計量を使って入力を正規化する
        if self.is_train:
            mu = x.mean(0).mean(2).mean(3)
            sigma = (((x - mu) ** 2).mean(0).mean(2).mean(3) + self.eps).sqrt()

            # 全体の統計量を更新する
            self.running_mu = self.alpha * self.running_mu \
                                + (1 - self.alpha) * mu.value
            self.running_sigma = self.alpha * self.running_sigma \
                                + (1 - self.alpha) * sigma.value

        # 推定時は移動平均の値を使って入力を正規化する
        else:
            mu = self.mu 
            sigma = self.sigma

        return self.parameters["W"] * ((x - mu) / sigma) + self.parameters["b"]

class RecurrentCell(Layer):
    """
    Elmanネットワークで使用されたリカレントユニットのレイヤー

    参考
        Elman, Jeffrey L. "Finding structure in time." Cognitive science 14.2 (1990): 179-211.
    """

    def __init__(self, num_in_units, num_h_units):
        """
        引数
            num_in_units: 前のレイヤーの隠れユニット数
            num_h_units: このレイヤーの隠れユニット数
        """

        super().__init__()

        # 隠れベクトルを初期化する際に使う
        self.num_h_units = num_h_units

        self.parameters = {
            "W": node.Node(np.random.randn(num_in_units, num_h_units) * np.sqrt(1. / num_in_units)),
            "U": node.Node(np.random.randn(num_h_units, num_h_units) * np.sqrt(1. / num_in_units)),
            "b": node.Node(np.zeros(num_h_units))
        }

    def reset(self):
        """
        隠れベクトルを零ベクトルで初期化する。
        """

        return node.Node(np.zeros(self.num_h_units))

    def __call__(self, x, h):

        y = x.dot(self.parameters["W"]) + h.dot(self.parameters["U"]) + self.parameters["b"] 

        return y.sigmoid()

class RecurrentLayer(RecurrentCell):

    def __call__(self, x, h):
        """
        引数
            x: 時系列データ
        """

        # 隠れベクトルを初期化する
        h = self.reset()

        seq = []
        for t in range(x.value.shape[1]):
            y = x[:, t].dot(self.parameters["W"]) + h.dot(self.parameters["U"]) + self.parameters["b"] 
            h = y.sigmoid()
            seq.append(h.expand(1))

        return cat(seq, axis=1) 

####################
###              ###
### Convolutions ###
###              ###
####################

class Conv2D(Layer):
    """
    各チャンネルの入力が2Dデータの場合の畳み込み演算
    """

    def __init__(self, num_in_ch, num_out_ch, filter_size, stride=1, pad=0):
        super(Conv2D, self).__init__()

        self.filter_size = filter_size
        self.stride = stride 
        self.pad = pad

        # Loweringされた行列との積をとるので、あらかじめ行列にしておく
        self.parameters = {
            "W": node.Node(np.random.randn(num_out_ch, num_in_ch, filter_size, filter_size)),
            "b": node.Node(np.zeros(num_out_ch))
        }

    def __call__(self, input):
        FN, _, _, _ = self.parameters["W"].value.shape
        hidden = input
        hidden = hidden.lower(self.filter_size, self.stride, self.pad)
        hidden = hidden.dot(self.parameters["W"].reshape(FN, -1).t()) + self.parameters["b"]

        # 行列を元の形に戻す
        N, C, H, W = input.value.shape
        output_size = 1 + int((H + 2 * self.pad - self.filter_size) / self.stride)
        return hidden.reshape(N, output_size, output_size, -1).transpose(0, 3, 1, 2)

class ConvTranspose2D(Layer):
    """
    畳み込み層のバックワード計算を使ったTransposed Convolution層
    """

    def __init__(self, num_in_ch, num_out_ch, filter_size, stride=1, pad=0):
        super(ConvTranspose2D, self).__init__()

        self.filter_size = filter_size
        self.stride = stride 
        self.pad = pad

        # 
        self.parameters = {
            "W": node.Node(np.random.randn(num_in_ch, num_out_ch, filter_size, filter_size) * 0.001)
        }

class MaxPool2D(Layer):
    """
    各チャンネルの入力が2Dデータの場合のMax Pooling演算
    """

    def __init__(self, filter_size, stride=1, pad=0):
        super(MaxPool2D, self).__init__()

        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad

        self.parameters = {}

    def __call__(self, input):
        hidden = input
        hidden = hidden.lower(self.filter_size, self.stride, self.pad)
        hidden = hidden.reshape(-1, self.filter_size**2)
        hidden = hidden.max(1)

        # 行列を元の形に戻す
        N, C, H, W = input.value.shape
        output_size = int(1 + (H - self.filter_size) / self.stride)
        return hidden.reshape(N, output_size, output_size, C).transpose(0, 3, 1, 2)