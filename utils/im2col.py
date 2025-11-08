import cupy as np
from typing import Tuple, Optional

class Im2Col:
    """
    A class to handle im2col and col2im operations for convolutional layers
    """
    
    @staticmethod
    def im2col(array: np.ndarray, filter_dim: Tuple[int, int], pad: int, stride: int) -> np.ndarray:
        """
        Transform image to columns for convolution operation
        
        Args:
            array: Input array of shape (batch_size, channels, height, width)
            filter_dim: (filter_height, filter_width)
            pad: Padding size
            stride: Stride size
            
        Returns:
            Column matrix of shape (channels*filter_h*filter_w, batch_size*output_h*output_w)
        """
        batch_size, channels, height, width = array.shape
        filter_h, filter_w = filter_dim
        
        # Calculate output dimensions
        height_out = (height + 2 * pad - filter_h) // stride + 1
        width_out = (width + 2 * pad - filter_w) // stride + 1
        
        # Pad the array if needed
        if pad > 0:
            array_padded = np.pad(array, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        else:
            array_padded = array
        
        # Initialize output columns
        cols = np.zeros((channels * filter_h * filter_w, batch_size * height_out * width_out))
        
        # Extract patches
        col_idx = 0
        for b in range(batch_size):
            for i in range(height_out):
                for j in range(width_out):
                    patch = array_padded[b, :, 
                                       i*stride:i*stride+filter_h, 
                                       j*stride:j*stride+filter_w]
                    cols[:, col_idx] = patch.flatten()
                    col_idx += 1
                    
        return cols
    
    @staticmethod
    def col2im(cols: np.ndarray, array_shape: Tuple, filter_dim: Tuple[int, int], 
               pad: int, stride: int) -> np.ndarray:
        """
        Transform columns back to image format
        
        Args:
            cols: Column matrix from im2col
            array_shape: Original array shape (batch_size, channels, height, width)
            filter_dim: (filter_height, filter_width)
            pad: Padding size used in im2col
            stride: Stride size used in im2col
            
        Returns:
            Reconstructed array of original shape
        """
        batch_size, channels, height, width = array_shape
        filter_h, filter_w = filter_dim
        
        height_out = (height + 2 * pad - filter_h) // stride + 1
        width_out = (width + 2 * pad - filter_w) // stride + 1
        
        array_padded = np.zeros((batch_size, channels, height + 2*pad, width + 2*pad))
        
        # Reconstruct from columns
        col_idx = 0
        for b in range(batch_size):
            for i in range(height_out):
                for j in range(width_out):
                    patch = cols[:, col_idx].reshape(channels, filter_h, filter_w)
                    array_padded[b, :, 
                               i*stride:i*stride+filter_h, 
                               j*stride:j*stride+filter_w] += patch
                    col_idx += 1
        
        # Remove padding
        if pad > 0:
            return array_padded[:, :, pad:-pad, pad:-pad]
        return array_padded

    @staticmethod
    def calculate_output_dim(input_dim: int, filter_dim: int, pad: int, stride: int) -> int:
        """Calculate output dimension size"""
        return (input_dim + 2 * pad - filter_dim) // stride + 1