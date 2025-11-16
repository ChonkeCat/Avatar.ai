from .layer import Layer
try:
    import cupy as cp
except ImportError:
    import numpy as cp

class Dropout(Layer):
    def __init__(self, rate, input_shape=None, first=False):
        super().__init__(input_shape=input_shape, activation_func=None, first=first)
        self.rate = rate
        self.mask = None

    def initialize(self, input_shape):
        self.output_shape = input_shape
        return self.output_shape

    def forward(self, inputs):
        if self.rate > 0:
            self.mask = cp.random.binomial(1, 1-self.rate, size=inputs.shape) / (1-self.rate)
            self.A = inputs * self.mask
        else:
            self.A = inputs
        return self.A

    def backward(self, gradient):
        if self.rate > 0:
            return gradient * self.mask
        else:
            return gradient