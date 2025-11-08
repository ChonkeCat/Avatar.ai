# the dense layer class
from .layer import Layer
from typing import Tuple
import cupy as cp # change is cupy if using gpu

class dense(Layer):
    def __init__(self, neurons: int, activation_func, input_shape: Tuple[int, int, int, int] = None, first = False):
        self.neurons = neurons
        super().__init__(input_shape = input_shape, activation_func = activation_func, first = first)

    def initialize(self, input_shape):
        input_length = input_shape[1]
        bias_shape = (1, self.neurons)
        self.out_shape = [input_shape[0], self.neurons]
        self.W = cp.random.randn(self.neurons, input_length)/cp.sqrt(input_length)
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
        pre_activation_wrt_activation_gradient = self.activation_func(self.A, grad=True)
        pre_activation_gradient = pre_activation_wrt_activation_gradient * gradient

        pre_activation_wrt_weights_gradient = self.A_prev
        batch = self.A_prev.shape[0]
        self.dW = cp.dot(pre_activation_gradient.T, pre_activation_wrt_weights_gradient)/batch
        self.db = pre_activation_gradient.sum(axis=0, keepdims=True)/batch
        prev_gradient = pre_activation_gradient.dot(self.W)
        return prev_gradient