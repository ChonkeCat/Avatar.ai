# the dense layer class
from model.layer import Layer
from typing import Tuple
import cupy as cp

class dense(Layer):
    def __init__(self, neurons: int, activation_func, input_shape: Tuple[int, int, int, int] = None, first = False):
        super.__init__(input_shape = input_shape, activation_func = activation_func, first = first)
        self.neurons = neurons

    def forward(self, A_prev):
        pass

    def backward(self, gradient):
        pass