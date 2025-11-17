# -----------------------------
# Use CuPy if available
# -----------------------------
try:
    import cupy as cp
    np = cp
except ImportError:
    import numpy as np
    cp = np

import torch
import torch.nn as nn
from model.Conv2D import Conv2D  # your custom layer

# -----------------------------
# Identity activation
# -----------------------------
def identity(x, grad=False):
    return x if not grad else np.ones_like(x)

# -----------------------------
# Seeds
# -----------------------------
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# Random input
# -----------------------------
N,H,W,C = 2,5,5,3
x_np = np.random.randn(N,H,W,C).astype(np.float32)

# -----------------------------
# PyTorch Conv2d (no padding)
# -----------------------------
in_channels = C
out_channels = 2
kernel_size = 3
stride = 1
padding = 0  # remove padding

conv_torch = nn.Conv2d(in_channels, out_channels, kernel_size,
                       stride=stride, padding=padding, bias=True)

# Random weights/bias
conv_torch.weight.data = torch.randn_like(conv_torch.weight.data)
conv_torch.bias.data = torch.randn_like(conv_torch.bias.data)

# Convert input to PyTorch format: (N,H,W,C) -> (N,C,H,W)
x_torch = torch.from_numpy(cp.asnumpy(x_np)).permute(0,3,1,2)
out_torch = conv_torch(x_torch)
out_torch_np = out_torch.permute(0,2,3,1).detach().numpy()

# -----------------------------
# Custom Conv2D (no padding)
# -----------------------------
conv_custom = Conv2D(filter_size=(kernel_size, kernel_size),
                     n_filters=out_channels,
                     activation_func=identity,
                     padding='valid',  # no padding
                     stride=stride)
conv_custom.initialize(input_shape=x_np.shape)

# Copy weights/bias (PyTorch: (out,in,H,W) -> custom: (H,W,C_in,C_out))
conv_custom.W[:] = cp.array(conv_torch.weight.data.permute(2,3,1,0).numpy())
conv_custom.b[:] = cp.array(conv_torch.bias.data.numpy())

# Forward pass
out_custom = conv_custom.forward(x_np)

# -----------------------------
# Compare outputs
# -----------------------------
print("PyTorch output shape:", out_torch_np.shape)
print("Custom Conv2D output shape:", out_custom.shape)

diff = cp.abs(out_custom - cp.array(out_torch_np))
print("Max absolute difference:", diff.max())
print("Mean absolute difference:", diff.mean())
