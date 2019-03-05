from collections import *
import numpy as np

"""
動的計算グラフを表すリスト。各要素はあるOpとNodeのペアで構成され、バックワード計算の際に後ろから順に誤差信号を伝播する。
"""
Pair = namedtuple("Pair", ("op", "node"))
TREE = []

# このフラグがTrueの時だけ計算ブラフを構築する。
CONSTRUCT_TREE = True
class zero_grad(object):
    """
    主に類推時に使われる。定義されるwith構文の中では計算グラフが構築されない。
    """

    def __enter__(self, *args):

        global CONSTRUCT_TREE
        CONSTRUCT_TREE = False

    def __exit__(self, *args):

        global CONSTRUCT_TREE
        CONSTRUCT_TREE = True

def _add_new_pair(op, node):
    """
    計算グラフにあらたなペアを追加する。

    引数
        op: Opインスタンス
        node: Nodeインスタンス
    """

    # 計算グラフを構築しないなら、すぐNoneを返す。
    if not CONSTRUCT_TREE:
        return

    global TREE 
    node = Pair(op, node)
    TREE.append(node)

def _destruct_tree():
    """
    計算グラフをリセットする。
    """

    global TREE 
    TREE = TREE.__new__(list)

def _core_scaler2node(x):
    """
    実数値をNodeインスタンスに変換する。

    引数
        x: 数値型
    """

    # パラメーターではなく、葉であるので勾配を計算する必要がない。
    return Node(np.array(x), no_grad=True)

def _scaler2node(fn):
    """
    引数を調べてどちらかが数値型なら、Nodeインスタンスにに変換してからfnで定義される演算を行う。

    引数
        fn: Opクラス
    """

    def _wrapper(x, y):

        # どちらかが数値型の場合、Nodeインスタンスに変換してからfnで定義される演算を行う。
        if type(x) != Node:
            x = _core_scaler2node(x)

        elif type(y) != Node:
            y = _core_scaler2node(y)

        return fn(x, y)

    return _wrapper

def _core_broadcast(x, y):
    """
    両Operand間で演算が定義できるように必要ならば形を変える。

    引数
        x, y: Nodeインスタンス
    """

    # 1.
    # スカラー x ベクトル(行列) 
    if len(x.value.shape) == 0:
        z = x.rep(None, y.value.shape, False)
        return z, y

    elif len(y.value.shape) == 0:
        z = y.rep(None, x.value.shape, False)
        return x, z

    # 2.
    # ベクトル x 行列
    if x.value.ndim == 1 and y.value.ndim == 2:

        # ベクトルの要素数が行列の列数と合わない場合、ValueErrorを発生させる。
        if x.value.shape[0] != y.value.shape[1]:
            print("The size of y should match one of the last dimension of x.")
            print("x shape: {}".format(x.value.shape))
            print("y shape: {}".format(y.value.shape))
            raise(ValueError)  

        z = x.rep(0, (y.value.shape[0], 1), False)
        return z, y

    elif x.value.ndim == 2 and y.value.ndim == 1:

        if x.value.shape[1] != y.value.shape[0]:
            print("The size of y should match one of the last dimension of x.")
            print("x shape: {}".format(x.value.shape))
            print("y shape: {}".format(y.value.shape))
            raise(ValueError)  

        z = y.rep(0, (x.value.shape[0], 1), False)
        return x, z

    # 3.
    # 行列 x 行列
    if x.value.ndim == 2 and y.value.ndim == 2:

        # 行数も列数も異なり、かつそれらが1でない場合、ValueErrorを発生させる。
        if x.value.shape[0] != y.value.shape[0] and x.value.shape[1] != y.value.shape[1]:
            if x.value.shape[0] != 1 and x.value.shape[1] != 1:
                raise(ValueError)
                
            elif y.value.shape[0] != 1 and y.value.shape[1] != 1:
                raise(ValueError)

        # xとyで列数が違う場合、多い方に合わせて少ない方をブロードキャストする。
        if x.value.shape[1] < y.value.shape[1]:
            z = x.rep(1, (1, y.value.shape[1]), True)
            return z, y

        elif x.value.shape[1] > y.value.shape[1]:
            z = y.rep(1, (1, x.value.shape[1]), True)
            return x, z

        # xとyで列数が違う場合、多い方に合わせて少ない方をブロードキャストする。 
        if x.value.shape[0] < y.value.shape[0]:
            z = x.rep(0, (y.value.shape[0], 1), True)
            return z, y

        else:
            z = y.rep(0, (x.value.shape[0], 1), True)
            return x, z

    # ここには到達しない。
    else:
        raise(ValueError)

    return x, y

def _broadcast(fn):
    """
    入力の形が異なる場合、一方に合わせて他方の形を変形する。
    """

    def _wrapper(x, y):

        # xとyをスワップすると、-など可換でないオペランドの場合にバグが発生するので、xとyを別々に考える必要がある。
        if x.value.shape != y.value.shape:
            x, y = _core_broadcast(x, y)

        return fn(x, y)

    return _wrapper

"""
2つの値を取る演算子のデコレーター：
2つの値を受け取り、fnで定義される演算子を適用する。値がスカラーである場合（Nodeオブジェクトのインスタンスではない）、それをNodeオブジェクトのインスタンスに変換したのち、必要な場合にブロードキャストすることで値間の次元を合わせる。次元と形が等しい2つのNodeオブジェクトのインスタンスが用意できたので、fnで定義される演算子を適用し、その出力で定義されるNodeオブジェクトのインスタンスとのペアを動的計算グラフに追加する。
"""
def _two_oprand_op(fn):
    def wrapper(x, y):
        op = fn(x, y)
        node = Node(op.output)
        _add_new_pair(op, node)
        return node
    return wrapper

"""
1つの値を取る演算子のデコレーター：
上の関数の値が1つの場合。
"""
def _single_oprand_op(fn):
    def wrapper(x, *args):
        # もしxがNodeオブジェクトのインスタンスではない場合、変換しておく。
        if type(x) != Node:
            x = _core_scaler2node(x)
        op = fn(x, *args)
        node = Node(op.output)
        _add_new_pair(op, node)
        return node
    return wrapper

def cat(x, axis=0):
    """
    Nodeインスタンスのリストを繋げる
    """

    op = Cat(x, axis)
    node = Node(op.output)
    _add_new_pair(op, node)

    return node

class Node(object):
    def __init__(self, value, no_grad=False):
        self.value = value
        self._no_grad = no_grad
        
        # no_gradがTrueの場合、このNodeオブジェクトのインスタンスは計算グラフの葉なので、勾配のプロパティを必要としない。
        if not self._no_grad:
            self.grad = np.zeros(self.value.shape)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __add__(self, x):
        return Add(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __radd__(self, x):
        return Add(x, self)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __sub__(self, x):
        return Sub(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rsub__(self, x):
        return Sub(x, self)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __mul__(self, x):
        return Mul(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rmul__(self, x):
        return Mul(x, self)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __truediv__(self, x):
        return Div(self, x)

    @_scaler2node
    @_broadcast
    @_two_oprand_op
    def __rtruediv__(self, x):
        return Div(x, self)

    @_two_oprand_op
    def dot(self, x):
        return Dot(self, x)

    @_single_oprand_op
    def mean(self, axis=0):
        return Mean(self, axis)

    @_single_oprand_op
    def sum(self, axis=1):
        return Sum(self, axis)

    @_single_oprand_op
    def __pow__(self, x):
        return Pow(self, x)

    @_single_oprand_op
    def exp(self):
        return Exp(self)

    @_single_oprand_op
    def log(self):
        return Log(self)

    @_single_oprand_op
    def sqrt(self):
        return Sqrt(self)

    @_single_oprand_op
    def __getitem__(self, idx):
        return GetItem(self, idx)

    @_single_oprand_op
    def gather(self, indeces, axis=1):
        return TakeAlongAxis(self, indeces, axis)

    @_single_oprand_op
    def rep(self, axis=0, times=1, keepdims=True):
        return Rep(self, axis, times, keepdims)

    @_single_oprand_op
    def clip(self, min_bound=0., max_bound=np.inf):
        return Clip(self, min_bound, max_bound)

    @_single_oprand_op
    def sign(self):
        return Sign(self)

    @_single_oprand_op
    def expand(self, axis=1):
        return Expand(self, axis)

    @_single_oprand_op
    def t(self):
        return T(self)

    @_single_oprand_op
    def transpose(self, *perm):
        return Transpose(self, perm)

    @_single_oprand_op
    def reshape(self, *shape):
        return Reshape(self, shape)

    @_single_oprand_op
    def sigmoid(self):
        return Sigmoid(self)

    @_single_oprand_op
    def tanh(self):
        return Tanh(self)

    @_single_oprand_op
    def relu(self):
        return ReLU(self)

    @_single_oprand_op
    def leaky_relu(self, alpha=0.2):
        return LeakyReLU(self, alpha)

    @_single_oprand_op
    def selu(self):
        return SeLU(self)

    @_single_oprand_op
    def softmax(self):
        return Softmax(self)

    @_two_oprand_op
    def binary_cross_entropy(self, x):
        return BinaryCrossEntropy(self, x)

    @_two_oprand_op
    def softmax_with_cross_entropy(self, x):
        return SoftmaxWithCrossEntropy(self, x)

    @_two_oprand_op
    def mean_squared_error(self, x):
        return MeanSquaredError(self, x)

    @_two_oprand_op
    def huber_loss(self, x):
        return HuberLoss(self, x)

    @_two_oprand_op
    def kl_divergence(self, x):
        return KLDivergence(self, x)

    def acc_grad(self, grad):
        if not self._no_grad:
            self.grad += grad

    def update(self, delta):
        self.value += delta

    def zero_grad(self):
        self.grad.fill(0.)

    def clear_tree(self):
        _destruct_tree()

    def backward(self, err_sig=None):
        for i, pair in enumerate(TREE[::-1]):
            if i == 0:
                if err_sig is not None:
                    pair.op.backward(err_sig)
                else:
                    pair.op.backward(np.ones(self.value.shape))
            else:
                pair.op.backward(pair.node.grad)

        _destruct_tree()

class Op(object):
    def __init__(self, *srcs):
        self._srcs = srcs

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

class Dot(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.output = np.dot(self._srcs[0].value, self._srcs[1].value)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(np.dot(err_sig, self._srcs[1].value.T))
        self._srcs[1].acc_grad(np.dot(self._srcs[0].value.T, err_sig))

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
        self.output = np.tile(self._srcs[0].value, self._args[1])

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

class Sign(Op):
    """
    フィードフォワード時の計算はnumpyのsign関数と同じだが、エラーシグナルが流れるようにするため、絶対値が1以下ならば、線形関数のように扱う。
    """

    def __init__(self, x, *args):

        super().__init__(x)

        self.output = np.sign(self._srcs[0].value)

    def backward(self, err_sig):

        return self._srcs[0].acc_grad(err_sig * 1 * (np.abs(self.output) <= 1))

class Expand(Op):
    """
    新たな次元を追加する。
    """

    def __init__(self, x, *args):

        super().__init__(x)

        self.axis = args[0]

        self.output = np.expand_dims(self._srcs[0].value, axis=args[0])

    def backward(self, err_sig):

        self._srcs[0].acc_grad(np.squeeze(err_sig, axis=self.axis))

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

class HuberLoss(Op):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self._a = self._srcs[0].value - self._srcs[1].value

        # ! _bはあとで使うわけではないが、一応保存しておく。
        self._b = np.where(np.abs(self._a)<1., 0.5*self._a**2, np.abs(self._a)-0.5)
        self.output = np.mean(np.sum(self._b, axis=1, keepdims=False), axis=0, keepdims=False)

    def backward(self, err_sig):
        self._srcs[0].acc_grad(np.where(np.abs(self._a)<1., self._a, np.sign(self._a)))

class KLDivergence(Op):
    """
    正規分布のパラメータ2つ(ただし、平均と*分散の対数*)を受け取り、その分布と平均0、分散1で表される正規分布とのKLダイバージェンスを解析的に計算する。主にVarietional Auto Encoder(VAE)の損失関数の一部として使われる。

    math:
        L(\mu, \log (\sigma^2)) = (\mu^2 + \sigma^2 - \log (\sigma^2) - 1) \cdot 0.5
    """

    def __init__(self, mu, ln_var, *args):

        super().__init__(mu, ln_var)
 
        self.output = -0.5 * np.mean(np.sum(1 + np.log(self._srcs[1].value**2) - self._srcs[0].value**2 - self._srcs[1].value**2, axis=1))

    def backward(self, err_sig):

        shape = [1 for _ in range(len(self._srcs[0].value.shape))]
        err_sig = np.tile(1, shape) / self._srcs[0].value.shape[0]

        self._srcs[0].acc_grad(err_sig * self._srcs[0].value)
        self._srcs[1].acc_grad(err_sig * (-self._srcs[1].value/self._srcs[1].value**2 + self._srcs[1].value))

