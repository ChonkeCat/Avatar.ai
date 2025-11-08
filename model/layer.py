#layer definition, skeleton class for layers
import cupy as cp # change is cupy if using gpu
from typing import Tuple

class Layer():
    def __init__(self, input_shape:Tuple[int, int, int, int]=None, activation_func=None, first:bool=False): #input shape is nhwc (batch size, height, width, channels)
        self.input_shape = input_shape
        self.activation_func = activation_func
        self.first = first
        if first and input_shape is None:
            raise ValueError("First layer must have input shape defined")
        elif input_shape is not None:
            self.initialize(input_shape=input_shape)
        #  basic fields
        self.A = None            # activated output (post-activation)
        self.Z = None            # pre-activation (linear output before activation)
        self.W = cp.zeros(1)     # weights
        self.dW = cp.zeros(1)    # gradient of weights
        self.mo = cp.zeros(1)    # momentum term for optimizer
        self.acc = cp.zeros(1)   # accumulator (e.g., for Adam)
        self.mo_b = cp.zeros(1)  # momentum for bias
        self.acc_b = cp.zeros(1) # accumulator for bias
        self.b = cp.zeros(1)     # biases
        self.db = cp.zeros(1)    # gradient of bias

    def initialize(self, input_shape): # will be defined in child classes
        pass

    def forward(self, input): # will be defined in child classes
        pass

    def backward(self, gradient): # will be defined in child classes
        pass