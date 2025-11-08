#model architecture definition, e.g. forward pass, etc.
import cupy as cp
import pickle

class model():
    def __init__(self):
        self.layers = []
        self.learning_rate_mask = []
        # add stats later

    def add(self, layer, lr_ratio=1):
        self.layers.append(layer)
        self.learning_rate_mask.append(lr_ratio)

    def insert(self, layer, lr_ratio, i):
        self.layers.insert(i, layer)
        self.learning_rate_mask.append(lr_ratio)

    def pop(self):
        self.layers = self.layers[:-1]

    def compile(self):
        output_shape = self.layers[0].initialize(self.layers[0].input_shape)
        for i in range(len(self.layers[1:])):
            output_shape = self.layers[i + 1].initialize(output_shape)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
    def update(self, learning_rate, beta1=0.9, beta2=0.99):
        for layer, lr_ratio in zip(self.layers, self.learning_rate_mask):
            actual_learning_rate = lr_ratio * learning_rate

            # Adam updates for weights
            layer.mo = beta1 * layer.mo + (1 - beta1) * layer.dW
            layer.acc = beta2 * layer.acc + (1 - beta2) * (layer.dW * layer.dW)
            layer.W -= actual_learning_rate * layer.mo / (cp.sqrt(layer.acc) + 1e-7)

            # Adam updates for biases
            layer.mo_b = beta1 * layer.mo_b + (1 - beta1) * layer.db
            layer.acc_b = beta2 * layer.acc_b + (1 - beta2) * (layer.db * layer.db)
            layer.b -= actual_learning_rate * layer.mo_b / (cp.sqrt(layer.acc_b) + 1e-7)

    def train(self, loss_func, x, y, epochs = 50, learning_rate = 0.001, decay = 0.96):
        pass

    def save(self, path):
        save = open(path, "wb")
        pickle.dump(self, save)
        save.close()

    def load(self, path):
        load = open(path, "rb")
        model = pickle.load(load)
        load.close()
        return model