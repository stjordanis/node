# DL勉強用のライブラリ

## 使い方

以下ではこれらのライブラリをインポートしていると仮定する。

~~~python
import node
import numpy as np
~~~

### 変数の作成方法

3x3の乱数行列変数xを作成する。

~~~python
x = node.Node(np.random.randn(3, 3))
~~~

### 変数が持つ値の確認方法

変数が持つ値を確認するにはvalueプロパティを使う。

~~~python
print(x.value)
~~~

### 演算

変数同士の演算は普通に行う。

~~~python
y = node.Node(np.random.randn(3, 3))
z = x + y
~~~

### ブロードキャスト

ブロードキャストのルールはnumpyと同じ。

~~~python
y = node.Node(np.random.randn(3))
z = x + y
~~~

## ニューラルネットワークの定義

新たなニューラルネットワークを定義するには、node.Networkクラスを継承して__call__メソッドを定義する。

~~~python
class Classifier(node.Network):

    def __init__(self, 
                 num_in_units, 
                 num_h_units,
                 num_out_units):
  
        # パラメーターを持つレイヤーはself.layersのリストに入れる
        self.layers = [
            node.Linear(num_in_units, num_h_units),
            node.Linear(num_h_units, num_out_units)
        ]

    def __call__(self, input):
    
        # フォワード計算を定義する。
        hidden = self.layers[0](input).tanh()
        output = self.layers[1](hidden)

        return output

classifier = Classifier(10, 50, 10)
~~~

### オプティマイザーの定義

オプティマイザーを定義するには最適化したいパラメーターのリストと学習率の値を渡す。

~~~python
optimizer = node.SGD(classifier.get_parameters(), 0.001)
~~~

### 損失関数

損失関数は活性化関数と同じように使う。

~~~python
input = node.Node(np.random.randn(1, 10))
target = node.Node(np.random.randn(1, 10))
prediction = classifier(input)
loss = prediction.softmax_with_binary_cross_entropy(target)
~~~

### バックワード計算

構築した計算グラフに対し、バックワード計算をしたい場合、変数のbackwardメソッドを呼ぶ。

~~~python
loss.backward()
~~~

### パラメーターのアップデート

バックワード計算で求めた勾配を使ってパラメータを更新したい場合、定義したオプティマイザーを呼ぶ。

~~~python
optimizer.update()
~~~
