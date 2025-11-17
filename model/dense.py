# the dense layer class
from .layer import Layer
from typing import Tuple
try:
    import cupy as cp
except ImportError:
    import numpy as cp

class dense(Layer):
    def __init__(self, neurons: int, activation_func, input_shape: Tuple[int, int, int, int] = None, first = False):
        self.neurons = neurons
        super().__init__(input_shape = input_shape, activation_func = activation_func, first = first)

    def initialize(self, input_shape):
        input_length = input_shape[1]
        bias_shape = (1, self.neurons)
        self.out_shape = [input_shape[0], self.neurons]
        fan_in = input_length
        self.W = cp.random.randn(self.neurons, input_length) * cp.sqrt(2.0 / fan_in)
        self.b = cp.full(bias_shape, 0.0, dtype=cp.float32)
        self.db = cp.full(bias_shape, 0.0, dtype=cp.float32)
        #optimizer stuff
        self.mo = cp.full(self.W.shape, 0.0, dtype=cp.float32)
        self.acc = cp.full(self.W.shape, 0.0, dtype=cp.float32)
        self.mo_b = cp.full(bias_shape, 0.0, dtype=cp.float32)
        self.acc_b = cp.full(bias_shape, 0.0, dtype=cp.float32)
        return self.out_shape

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = cp.dot(A_prev, self.W.T) + self.b
        self.A = self.activation_func(self.Z)
        return self.A

    def backward(self, gradient):
        batch = self.A_prev.shape[0]
        
        # Derivative of activation
        dZ = self.activation_func(self.Z, grad=True) * gradient
        
        # Gradients wrt weights
        self.dW = cp.dot(dZ.T, self.A_prev) / batch
        self.db = cp.sum(dZ, axis=0, keepdims=True) / batch
        
        # Gradient to pass to previous layer
        dA_prev = cp.dot(dZ, self.W)
        return dA_prev