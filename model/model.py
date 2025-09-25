#model architecture definition, e.g. forward pass, etc.
import cupy as cp

class model():
    def __init__(self):
        self.layers = []
        self.learning_rate_mask = []
        # add stats later

    def add(self, layer, lr_ratio):
        self.layers.append(layer)
        self.learning_rate_mask.append(lr_ratio)

    def insert(self, layer, lr_ratio, i):
        self.layers.insert(i, layer)
        self.learning_rate_mask.append(lr_ratio)

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
    
    def update(self, learning_rate, beta1=0.9, beta2=0.99):
        for i in range(len(self.layers)):
            actual_learning_rate = self.learning_rate_mask[i] * learning_rate
            # adam optimization
            self.Layers[i].mo = self.Layers[i].mo * beta1 + (1-beta1) * self.Layers[i].dW
            self.Layers[i].acc = beta2*self.Layers[i].acc  + (1-beta2) * (self.Layers[i].dW*self.Layers[i].dW)
            self.Layers[i].W = self.Layers[i].W - actual_learning_rate * self.Layers[i].mo/(cp.sqrt(self.Layers[i].acc) + 1e-07)   

            self.Layers[i].mo_b = self.Layers[i].mo_b*beta1 + (1-beta1)*self.Layers[i].db
            self.Layers[i].acc_b = beta2 * self.Layers[i].acc_b  + (1-beta2) * (self.Layers[i].db*self.Layers[i].db)
            self.Layers[i].b = self.Layers[i].b - actual_learning_rate * self.Layers[i].mo_b/(cp.sqrt(self.Layers[i].acc_b) + 1e-07)