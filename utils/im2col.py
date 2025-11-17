try:
    import cupy as np
except ImportError:
    import numpy as np
from typing import Tuple, Optional

class Im2Col:
    """
    A class to handle im2col and col2im operations for convolutional layers
    """
    
    @staticmethod
    def get_indices(in_shape, filter_h, filter_w, stride, padding):
        """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d. 
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    
        # get input size
        batch_size, channels, n_height, n_width = in_shape

        # get output size
        output_height = int((n_height + (2 * padding) - filter_h))//stride + 1
        output_width = int((n_width + (2 * padding) - filter_w))//stride + 1

        
        # Compute i matrix
        
        level_one_i = np.repeat(np.arange(filter_h),filter_w)
        level_one_i = np.tile(level_one_i, channels)
        
        every_level = stride * np.repeat(np.arange(output_height),output_width)
        
        i = level_one_i.reshape(-1,1) + every_level.reshape(1,-1)
        
        
        # compute j matrix
        
        level_one_j = np.tile(np.arange(filter_w), filter_h)
        level_one_j = np.tile(level_one_j, channels)
        
        every_level_j = stride * np.tile(np.arange(output_width), output_height)
        
        j = level_one_j.reshape(-1, 1) + every_level_j.reshape(1, -1)
        
        # compute d matrix
        d =  np.repeat(np.arange(channels),filter_h * filter_w).reshape(-1,1)
        
        return i,j,d
        
    @staticmethod
    def im2col(X, HF, WF, stride, pad):
        """
            Transforms our input image into a matrix.

            Parameters:
            - X: input image.
            - HF: filter height.
            - WF: filter width.
            - stride: stride value.
            - pad: padding value.

            Returns:
            -cols: output matrix.
        """
        N, C, H, W = X.shape
        # Padding
        X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = Im2Col.get_indices(X.shape, HF, WF, stride, pad)
        # Multi-dimensional arrays indexing.
        cols = X_padded[:, d, i, j]
        cols = cols.transpose(1, 2, 0).reshape(C*HF*WF, -1)
        return cols
        
    

    @staticmethod
    def col2im(dX_col, X_shape, HF, WF, stride, pad):
        N, C, H, W = X_shape
        H_padded, W_padded = H + 2*pad, W + 2*pad
        X_padded = np.zeros((N, C, H_padded, W_padded))

        i, j, d = Im2Col.get_indices(X_shape, HF, WF, stride, pad)

        # Reshape dX_col to (C*HF*WF, out_h*out_w, N)
        k, n_cols = dX_col.shape
        out_h = (H + 2*pad - HF)//stride + 1
        out_w = (W + 2*pad - WF)//stride + 1
        dX_col_reshaped = dX_col.reshape(C*HF*WF, out_h*out_w, N)
        dX_col_reshaped = dX_col_reshaped.transpose(2, 0, 1)  # (N, k, out_h*out_w)

        # Scatter back to padded image
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)

        # Remove padding
        if pad == 0:
            return X_padded
        return X_padded[:, :, pad:-pad, pad:-pad]



    @staticmethod
    def calculate_output_dim(input_dim: int, filter_dim: int, pad: int, stride: int) -> int:
        """Calculate output dimension size"""
        return (input_dim + 2 * pad - filter_dim) // stride + 1