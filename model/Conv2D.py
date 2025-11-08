# the 2D convolution layer class
from model.layer import Layer
from typing import Tuple
import cupy as np  # change to cupy later *Jerry
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
        self.kernel_size = filter_size  # Fixed typo: kernal_size -> kernel_size
        self.padding = padding  # Removed duplicate stride assignment
        
        self.cache = {}
        super().__init__(input_shape=input_shape,\
                    activation_func=activation_func,\
                    first=first)
        
        
    # example: conv_layer = Conv2D(filter_size=(3, 3), n_filters=32, ...)
    # real application: Conv2D((3, 3), 64, LeakyRelU, padding='same', input_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), first=True)
    def initialize(self, input_shape=None):
        self.batch_size, self.height_in, self.width_in, self.depth = input_shape
        
        '''
        1. take random values from standard normal distribution, then creates a 4D TENSOR made up of
        3x3 matrices of depth n_filters (e.g 32)
        ex: [[ 0.11, -0.11,  0.22], 
             [ 0.05, -0.08,  0.15],     
             [ -0.09, 0.12, -0.04]]

        '''
        # print(input_shape)
        # make a matrix of filter_size and then divide everything by its size to scale down
        self.W = np.random.randn(self.filter_size[0], self.filter_size[1],
            self.depth, self.n_filters)/(self.filter_size[0]*self.filter_size[1])
        
        # create bias 1D matrix and multiplies by 0.01 to scale down
        self.b = np.random.randn(self.n_filters) * 0.01

        # setting up values (set everythig to 0.0)
        #momentum, accumulator, gradient (derivative wrt Weight)
        self.momentum = np.full(self.W.shape, 0.0)
        self.accumulator = np.full(self.W.shape, 0.0)
        self.dW = np.full(self.W.shape, 0.0)
        self.momentum_b= np.zeros(self.b.shape)
        self.accumulator_b = np.zeros(self.b.shape)

        filter_h = self.W.shape[0]
        filter_w = self.W.shape[1]
        self.n_filters = self.W.shape[3]
        
        
        # same = with 0 padding, valid = no 0 padding
        
        # shape_after_conv calculates how big output is after filters (with/without padding)
        if self.padding == "same":
            self.shape_after_conv = (self.batch_size, self.height_in, self.width_in, self.n_filters)
            return self.shape_after_conv
        elif self.padding == "valid":
            height_out = ((self.height_in - filter_h) // self.stride) + 1
            width_out = ((self.width_in - filter_w) // self.stride) + 1
            
            # FIXED: Use the calculated output dimensions instead of input dimensions
            self.shape_after_conv = (self.batch_size, height_out, width_out, self.n_filters)
            return self.shape_after_conv
        else:
            raise Exception("Not a valid padding value")


    def forward(self, A_prev):
        self.inputs = A_prev
        self.A_prev = np.array(A_prev, copy=True)
        n = A_prev.shape[0]
        height_out = self.shape_after_conv[1]
        width_out = self.shape_after_conv[2]

        filter_h, filter_w, _, n_f = self.W.shape
        pad = self.calculate_pad_dims()
        w = np.transpose(self.W, (3, 2, 0, 1))

        self.cols = Im2Col.im2col(
            array=np.moveaxis(A_prev, -1, 1),
            filter_dim=(filter_h, filter_w),
            pad=pad[0],
            stride=self.stride
        )
        result = w.reshape((n_f, -1)).dot(self.cols)
        output = result.reshape(n_f, height_out, width_out, n)
        self.Z = output.transpose(3, 1, 2, 0) + self.b
        self.A = self.activation_func(self.Z)
        return self.A
            
    
    
    def backward(self, dLdA):
        batch_size = dLdA.shape[0]

        filter_h, filter_w, _, n_f = self.W.shape
        pad = self.calculate_pad_dims()

        # bias gradient calculation
        self.db = dLdA.sum(axis=(0, 1, 2)) / batch_size
        
        # dLdA reshapes to prep for matrix stuff
        da_curr_reshaped = dLdA.transpose(3, 1, 2, 0).reshape(n_f, -1)

        w = np.transpose(self.W, (3, 2, 0, 1))
        dw = da_curr_reshaped.dot(self.cols.T).reshape(w.shape)
        self.dW = np.transpose(dw, (2, 3, 1, 0))/batch_size

        output_cols = w.reshape(n_f, -1).T.dot(da_curr_reshaped)

        output = Im2Col.col2im(
            cols=output_cols,
            array_shape=np.moveaxis(self.A_prev, -1, 1).shape,
            filter_dim=(filter_h, filter_w),
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