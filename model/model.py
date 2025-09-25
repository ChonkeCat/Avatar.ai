#model architecture definition, e.g. forward pass, etc.
import cupy as cp

class model():
    def __init__(self):
        self.layers = []
        # add stats later
        # added optimizations (train mask etc)

    def add(self, layer):
        self.layers.append(layer)

    def insert(self, layer, i):
        self.layers.insert(i, layer)

    def pop(self):
        self.layers = self.layers[:-1]

    def compile(self):
        output_shape = self.layers[0].initialize(self.Layers[0].input_shape)
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


    
