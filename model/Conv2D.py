# the 2D convolution layer class
from model.layer import Layer
from typing import Tuple
try:
    import cupy as np
except ImportError:
    import numpy as np
from utils.im2col import Im2Col

class Conv2D(Layer):
    
    def __init__ (
        self,
        filter_size,
        n_filters,
        activation_func,
        b=0,
        input_shape=None,
        first=False,
        padding="valid",
        stride = 1):
        
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.b = b
        self.stride = stride
        self.kernel_size = filter_size
        self.padding = padding
        
        self.cache = {}
        super().__init__(input_shape=input_shape,\
                    activation_func=activation_func,\
                    first=first)
        
    def initialize(self, input_shape=None):
        self.batch_size, self.height_in, self.width_in, self.depth = input_shape
        
        fan_in = self.filter_size[0] * self.filter_size[1] * self.depth
        fan_out = self.filter_size[0] * self.filter_size[1] * self.n_filters
        
        # Use He initialization for LeakyReLU (better than Xavier for ReLU-like activations)
        std = np.sqrt(2.0 / fan_in)
        
        self.W = np.random.randn(
            self.filter_size[0], self.filter_size[1], self.depth, self.n_filters
        ).astype(np.float32) * std

        # Zero bias initialization
        self.b = np.zeros(self.n_filters, dtype=np.float32)

        # Use 'mo' and 'acc' for Adam optimizer
        self.mo = np.zeros(self.W.shape, dtype=np.float32)
        self.acc = np.zeros(self.W.shape, dtype=np.float32)
        self.dW = np.zeros(self.W.shape, dtype=np.float32)
        self.mo_b = np.zeros(self.b.shape, dtype=np.float32)
        self.acc_b = np.zeros(self.b.shape, dtype=np.float32)
        self.db = np.zeros(self.b.shape, dtype=np.float32)

        filter_h = self.W.shape[0]
        filter_w = self.W.shape[1]
        self.n_filters = self.W.shape[3]
        
        if self.padding == "same":
            self.shape_after_conv = (self.batch_size, self.height_in, self.width_in, self.n_filters)
            return self.shape_after_conv
        elif self.padding == "valid":
            height_out = ((self.height_in - filter_h) // self.stride) + 1
            width_out = ((self.width_in - filter_w) // self.stride) + 1
            
            self.shape_after_conv = (self.batch_size, height_out, width_out, self.n_filters)
            return self.shape_after_conv
        else:
            raise Exception("Not a valid padding value")


    def forward(self, A_prev):
        self.inputs = A_prev
        self.A_prev = A_prev
        batch_size = A_prev.shape[0]
        height_out = self.shape_after_conv[1]
        width_out = self.shape_after_conv[2]

        filter_h, filter_w, _, n_f = self.W.shape
        pad = self.calculate_pad_dims()
        w = np.transpose(self.W, (3, 0, 1, 2))

        self.cols = Im2Col.im2col(
            X=np.moveaxis(A_prev, -1, 1),
            HF = filter_h,
            WF=filter_w,
            pad=pad[0],
            stride=self.stride
        )
        
        # Multiply flattened filters with columns
        result = w.reshape(n_f, -1).dot(self.cols)

        # Reshape to match NHWC output
        output = result.reshape(n_f, height_out, width_out, batch_size)
        self.Z = output.transpose(3, 1, 2, 0) + self.b
        self.A = self.activation_func(self.Z)
        return self.A
            
    
    def backward(self, dLdA):
        batch_size = dLdA.shape[0]
        filter_h, filter_w, _, n_f = self.W.shape
        pad = self.calculate_pad_dims()

        # Gradient w.r.t Z (include activation derivative)
        dLdZ = self.activation_func(self.Z, grad=True) * dLdA
        
        # Gradient w.r.t bias - sum over batch, height, width
        self.db = dLdZ.sum(axis=(0, 1, 2))

        # Reshape dLdZ for convolution
        dLdZ_reshaped = dLdZ.transpose(3, 1, 2, 0)  # (n_filters, height_out, width_out, batch)
        dLdZ_flattened = dLdZ_reshaped.reshape(n_f, -1)

        # Use same W ordering as forward
        w = np.transpose(self.W, (3, 0, 1, 2))

        # Compute gradient w.r.t weights
        dw = dLdZ_flattened.dot(self.cols.T)
        dw = dw.reshape(w.shape)
        self.dW = np.transpose(dw, (1, 2, 3, 0))

        # Gradient w.r.t input - FIX: Create fresh array instead of accumulating
        output_cols = w.reshape(n_f, -1).T.dot(dLdZ_flattened)
        
        # CRITICAL FIX: Use the fixed col2im that doesn't accumulate
        output = Im2Col.col2im_fixed(
            dX_col=output_cols,
            X_shape=np.moveaxis(self.A_prev, -1, 1).shape,
            HF=filter_h,
            WF=filter_w,
            pad=pad[0],
            stride=self.stride
        )
        return np.transpose(output, (0, 2, 3, 1))

    def calculate_pad_dims(self) -> Tuple[int, int]:
        if self.padding == 'same':
            filter_h, filter_w, _, _ = self.W.shape
            return (filter_h - 1) // 2, (filter_w - 1) // 2
        elif self.padding == 'valid':
            return 0, 0
        else:
            raise Exception(f"Unsupported padding value: {self.padding}")

    @staticmethod
    def pad(array: np.array, pad: Tuple[int, int]):
        return np.pad(
            array=array,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant'
        )