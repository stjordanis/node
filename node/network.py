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
            if hasattr(layer, "parameters"):
                parameters += layer.get_parameters()

        # パラメータ数を記録する
        num_parameters = 0
        for parameter in parameters:
            num_parameters += parameter.value.size
        print("Has {} parameters".format(num_parameters))
                
        return parameters