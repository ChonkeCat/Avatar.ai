# the pooling layer class
from .layer import Layer
from typing import Tuple
try:
    import cupy as cp
except ImportError:
    import numpy as cp

class Pool(Layer):
    def __init__(self, pool_size: Tuple[int, int], stride: int = 2, input_shape: Tuple[int, int, int] = None, activation: str = None, first: bool = False):
        super().__init__(input_shape=input_shape, activation_func=activation, first=first)
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}

    def initialize(self, input_shape):
        batch_size, height, width, channels = input_shape
        h_poll, w_poll = self.pool_size
        h_out = (height - h_poll) // self.stride + 1
        w_out = (width - w_poll) // self.stride + 1
        self.output_shape = (batch_size, h_out, w_out, channels)
        return self.output_shape

    def forward(self, A_prev):
        self.A_prev = A_prev
        batch_size, h_out, w_out, channels = self.output_shape
        h_size, w_size = self.pool_size
        output = cp.zeros((batch_size, h_out, w_out, channels))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_size
                w_start = j * self.stride
                w_end = w_start + w_size

                slice_A_prev = self.A_prev[:, h_start:h_end, w_start:w_end, :]
                self.save_mask(slice=slice_A_prev, cords=(i, j))
                output[:, i, j, :] = cp.max(slice_A_prev, axis=(1, 2))
        self.A = output
        return self.A

    def backward(self, gradient):
        self.A = cp.zeros_like(self.A_prev)
        h_out = gradient.shape[1]
        w_out = gradient.shape[2]
        h_size, w_size = self.pool_size

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_size
                w_start = j * self.stride
                w_end = w_start + w_size
                self.A[:, h_start:h_end, w_start:w_end, :] += gradient[:, i:i + 1, j:j + 1, :] * self.cache[(i, j)]
                
        return self.A

    def save_mask(self, slice, cords):
        mask = cp.zeros_like(slice)
        batch_size, h, w, depth = slice.shape
        slice = slice.reshape(batch_size, h * w, depth)
        idmax = cp.argmax(slice, axis=1)

        batch_id, depth_id = cp.indices((batch_size, depth))
        mask.reshape(batch_size, h * w, depth)[batch_id, idmax, depth_id] = 1
        self.cache[cords] = mask