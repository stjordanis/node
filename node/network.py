class Network(object):

    def __init__(self):
        self.is_train = True

    def train(self):
        for layer in self.layers:
            layer.is_train = True 

    def test(self):
        for layer in self.layers:
            layer.is_train = False

    def get_parameters(self):
        parameters = []
        for layer in self.layers: 
            parameters += layer.get_parameters()

        # パラメータ数を記録する
        self.num_parameters = 0
        for parameter in parameters:
            self.num_parameters += parameter.value.size
                
        return parameters

    def get_num_parameters(self):
        return self.num_parameters