try:
    import cupy as cp
except ImportError:
    import numpy as cp

from model.model import model
from model.dense import dense
from model.Conv2D import Conv2D
from model.flatten import flatten
from model.pool import Pool
from model.dropout import Dropout
from utils.cnn_utils import LeakyRelU, softmax, crossentropyloss, process_dataset
import random

# Load data
print("Loading dataset...")
data_path = "C:/Projects/CNN/data_resized"
images, labels = process_dataset(data_path, augment=False)

images = cp.array(images, dtype=cp.float32)
labels = cp.array(labels, dtype=cp.float32)

dataset = list(zip(images, labels))
random.shuffle(dataset)

# Take enough for 5 batches
x_subset, y_subset = zip(*dataset[:160])
x_all = cp.array(x_subset, dtype=cp.float32)
y_all = cp.array(y_subset, dtype=cp.float32)

# Build model
batch_size = 32
net = model()
net.add(Conv2D((5, 5), 32, LeakyRelU, padding='same', input_shape=(batch_size, 80, 80, 3), first=True))
net.add(Pool((2, 2), 2))
net.add(Conv2D((3, 3), 64, LeakyRelU, padding='same'))
net.add(Pool((2, 2), 2))
net.add(Conv2D((3, 3), 96, LeakyRelU, padding='same'))
net.add(flatten())
net.add(dense(256, LeakyRelU))
net.add(Dropout(0.2))
net.add(dense(128, LeakyRelU))
net.add(Dropout(0.2))
net.add(dense(4, softmax))
net.compile()

print("\n" + "="*70)
print("TESTING GRADIENT ACCUMULATION ACROSS BATCHES")
print("="*70)

for batch_idx in range(5):
    print(f"\n--- BATCH {batch_idx + 1} ---")
    
    # Get batch
    start = batch_idx * 32
    end = start + 32
    x_batch = x_all[start:end]
    y_batch = y_all[start:end]
    
    # Check gradients BEFORE zeroing
    if batch_idx > 0:
        total_grad_before = sum(float(cp.linalg.norm(layer.dW)) for layer in net.layers 
                               if hasattr(layer, 'dW') and layer.dW.size > 1)
        print(f"Gradient norm BEFORE zeroing: {total_grad_before:.4f}")
    
    # Zero gradients (simulate what training loop does)
    for layer in net.layers:
        if hasattr(layer, 'dW') and layer.dW.size > 1:
            layer.dW[:] = 0
        if hasattr(layer, 'db') and layer.db.size > 1:
            layer.db[:] = 0
    
    # Check gradients AFTER zeroing
    total_grad_after_zero = sum(float(cp.linalg.norm(layer.dW)) for layer in net.layers 
                                if hasattr(layer, 'dW') and layer.dW.size > 1)
    print(f"Gradient norm AFTER zeroing: {total_grad_after_zero:.4f}")
    
    # Forward
    pred = net.forward(x_batch)
    loss = crossentropyloss(y_batch, pred, grad=False)
    
    # Backward
    grad = crossentropyloss(y_batch, pred, grad=True)
    net.backward(grad)
    
    # Check gradients AFTER backward
    total_grad_after_backward = sum(float(cp.linalg.norm(layer.dW)) for layer in net.layers 
                                    if hasattr(layer, 'dW') and layer.dW.size > 1)
    print(f"Loss: {loss:.4f}")
    print(f"Gradient norm AFTER backward: {total_grad_after_backward:.4f}")
    
    # Show individual layer gradients
    print("Individual layer gradients:")
    for i, layer in enumerate(net.layers):
        if hasattr(layer, 'dW') and layer.dW.size > 1:
            norm = float(cp.linalg.norm(layer.dW))
            print(f"  Layer {i:2d} ({layer.__class__.__name__:10s}): {norm:10.4f}")
    
    # Simulate update (with averaging and clipping)
    print("\nSimulating update...")
    
    # Average by batch
    for layer in net.layers:
        if hasattr(layer, 'dW') and layer.dW.size > 1:
            layer.dW /= batch_size
        if hasattr(layer, 'db') and layer.db.size > 1:
            layer.db /= batch_size
    
    total_grad_after_avg = sum(float(cp.linalg.norm(layer.dW)) for layer in net.layers 
                               if hasattr(layer, 'dW') and layer.dW.size > 1)
    print(f"Gradient norm AFTER averaging: {total_grad_after_avg:.4f}")
    
    # Clip
    net.clip_gradients(max_norm=5.0)
    
    total_grad_after_clip = sum(float(cp.linalg.norm(layer.dW)) for layer in net.layers 
                                if hasattr(layer, 'dW') and layer.dW.size > 1)
    print(f"Gradient norm AFTER clipping: {total_grad_after_clip:.4f}")

print("\n" + "="*70)
print("If gradients keep growing, there's accumulation happening somewhere!")
print("="*70)