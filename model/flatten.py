# the flatten layer class
from .layer import Layer
from typing import Tuple
try:
    import cupy as cp
except ImportError:
    import numpy as cp
import math

class flatten(Layer):
    def __init__(self, input_shape: Tuple[int, int, int, int] = None, activation_func=None, first = False):
        super().__init__(input_shape = input_shape, activation_func = activation_func, first = first)

    def initialize(self, input_shape):
        batch_size = input_shape[0]
        flattened_size = math.prod(input_shape[1:])
        self.out_shape = (batch_size, flattened_size)
        
        # Flatten layers don't have trainable parameters
        self.W = cp.array([])
        self.dW = cp.array([])
        self.b = cp.array([])
        self.db = cp.array([])
        self.mo = cp.array([])
        self.acc = cp.array([])
        self.mo_b = cp.array([])
        self.acc_b = cp.array([])
        
        return self.out_shape
    
    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = A_prev.reshape(A_prev.shape[0], -1)
        if self.activation_func:
            self.A = self.activation_func(self.Z, grad = False)
            return self.A
        self.A = self.Z
        return self.A
    
    def backward(self, gradient):
        if self.activation_func:
            activation_gradient = self.activation_func(self.Z, grad = True) * gradient
            gradient_prev = activation_gradient.reshape(self.A_prev.shape)
            return gradient_prev
        gradient_prev = gradient.reshape(self.A_prev.shape)
        return gradient_prev