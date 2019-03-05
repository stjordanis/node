class Network(object):

    def __init__(self):

        # このフラグが立っている場合、ドロップアウトやバッチ正則化が訓練時の動きをするようになる。
        self.is_train = True

    def training(self):
        """
        各レイヤーを訓練モードする
        """

        for layer in self.layers:
            layer.is_train = True 

    def evaluation(self):
        """
        各レイヤーを推論モードする
        """

        for layer in self.layers:
            layer.is_train = False

    def get_parameters(self):
        """
        各レイヤーのパラメーターのリストを返す
        """

        parameters = []
        for layer in self.layers: 
            parameters += layer.get_parameters()

        return parameters