from .node import Node, cat
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
            "W": Node(np.random.randn(num_in_units, num_h_units) * np.sqrt(1. / num_in_units)),
            "b": Node(np.zeros(num_h_units))
        }

    def __call__(self, x):

        return x.dot(self.parameters["W"]) + self.parameters["b"]

class BatchNorm(Layer):
    """
    バッチ正則化レイヤー

    参考
    https://arxiv.org/abs/1502.03167 (オリジナル)
    https://arxiv.org/abs/1702.03275
    """

    def __init__(self, num_in_units, alpha=0.999):
        """
        num_in_units := 前のレイヤーのユニット数
        alpha := 移動平均の更新率をコントロールするハイパーパラメーター
        """

        super(BatchNorm, self).__init__()

        self.parameters = {
            "W": Node(np.random.randn(num_in_units) * np.sqrt(1. / num_in_units)),
            "b": Node(np.zeros(num_in_units))
        }

        self.alpha = alpha

        # 標準偏差で除算する際にゼロ除算エラーが発生しないようにする
        self.eps = 1e-8

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

class AffineCoupling(Layer):

    def __init__(self, num_in_units, mask, s, t):

        super().__init__()
        
        # 入力に掛け合わせるマスク
        self.mask = mask 

        # スケーリングとトレンスレーティングを行うニューラルネット
        self.s = s 
        self.t = t

        # 対数尤度を計算するために行列式の値を記録しておく
        self.jacobian = None

    def invert(self, z):

        x = self.mask * z + (1 - self.mask) * ((z - self.t(self.mask * z)) * (-1 * self.s(self.mask * z)).exp())

        return x

    def __call__(self, x):

        y = self.s(self.mask * x)
        z = self.mask * x + (1 - self.mask) * (x * y.exp() + self.t(self.mask * x))
        
        # ヤコビアンを計算する
        self.jacobian = (((1 - self.mask) * y).sum(1)).exp()

        return z  

class CenterLoss(Layer):
    """
    "A Discriminative Feature Learning Approach for Deep Face Recognition"で提案された損失関数
    """

    def __init__(self, num_in_units, class_num, alpha=0.01, lam=1.0):
        """
        引数
            class_num: クラスターの数
            lam: この関数の貢献度(大きいほどクラスターの分散が小さくなる)
        """

        self.num_in_units = num_in_units
        self.class_num = class_num
        self.alpha = alpha
        self.lam = lam 

        self.clusters = [np.zeros(num_in_units) for _ in range(class_num)]

    def __call__(self, x, y):

        # フィードフォワード演算
        t = []
        for i in range(y.value.shape[0]):
            t.append(self.clusters[np.argmax(y.value[i])])
        z = x.mean_squared_error(Node(np.array(t)))

        # クラスターを更新分を求める
        delta = [np.zeros(self.num_in_units) for _ in range(self.class_num)]
        counter = [0 for _ in range(self.class_num)]
        for i in range(y.value.shape[0]):
            idx = np.argmax(y.value[i])
            delta[idx] += self.clusters[idx] - x.value[i] 
            counter[idx] += 1
        
        # 求めた更新分を使ってクラスターの値を更新する
        for i in range(self.class_num):
            self.clusters[i] -= self.alpha * (delta[i] / (1 + counter[i]))

        return self.lam * z

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
            "W": Node(np.random.randn(num_in_units, num_h_units) * np.sqrt(1. / num_in_units)),
            "U": Node(np.random.randn(num_h_units, num_h_units) * np.sqrt(1. / num_in_units)),
            "b": Node(np.zeros(num_h_units))
        }

    def reset(self):
        """
        隠れベクトルを零ベクトルで初期化する。
        """

        return Node(np.zeros(self.num_h_units))

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