try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

print("=" * 70)
print(f"Backend: {'CuPy (GPU)' if GPU_AVAILABLE else 'NumPy (CPU)'}")
print("=" * 70)

import random
from model.model import model
from model.dense import dense
from model.Conv2D import Conv2D
from model.flatten import flatten
from model.pool import Pool
from model.dropout import Dropout
from utils.cnn_utils import LeakyRelU, softmax, crossentropyloss, process_dataset

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "C:/Projects/CNN/data_resized"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0005
DECAY = 0.97
USE_AUGMENTATION = True  # Enable data augmentation
USE_FULL_DATASET = True  # Use all data

print(f"\nConfiguration:")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  LR Decay: {DECAY}")
print(f"  Augmentation: {USE_AUGMENTATION}")
print(f"  Full Dataset: {USE_FULL_DATASET}")

# ============================================================
# LOAD DATASET
# ============================================================
print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

images, labels = process_dataset(DATA_PATH, augment=USE_AUGMENTATION)

# Convert to arrays
images = cp.array(images, dtype=cp.float32)
labels = cp.array(labels, dtype=cp.float32)

print(f"✓ Dataset loaded: {len(images)} images")
print(f"  Image shape: {images[0].shape}")
print(f"  Data stats - Mean: {images.mean():.4f}, Std: {images.std():.4f}")
print(f"  Data range - Min: {images.min():.4f}, Max: {images.max():.4f}")

# Verify data quality
if cp.any(cp.isnan(images)) or cp.any(cp.isinf(images)):
    raise ValueError("ERROR: NaN or Inf detected in input data!")

# Prepare dataset
dataset = list(zip(images, labels))
random.shuffle(dataset)

# Use subset or full dataset
if USE_FULL_DATASET:
    subset = dataset
    print(f"✓ Using FULL dataset ({len(subset)} images)")
else:
    subset_size = 2000
    subset = dataset[:subset_size]
    print(f"✓ Using subset ({subset_size} images)")

x_subset, y_subset = zip(*subset)
x_subset = list(x_subset)
y_subset = list(y_subset)

num_classes = y_subset[0].shape[0]
print(f"✓ Number of classes: {num_classes}")

# ============================================================
# BUILD MODEL
# ============================================================
print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

net = model()

# Layer 1: Conv + Pool (80x80x3 -> 40x40x32)
net.add(
    Conv2D((5, 5), 32, LeakyRelU, padding='same', 
           input_shape=(BATCH_SIZE, 80, 80, 3), first=True), 
    lr_ratio=0.5
)
net.add(Pool((2, 2), 2))

# Layer 2: Conv + Pool (40x40x32 -> 20x20x64)
net.add(Conv2D((3, 3), 64, LeakyRelU, padding='same'), lr_ratio=0.7)
net.add(Pool((2, 2), 2))

# Layer 3: Conv + Pool (20x20x64 -> 10x10x96)
net.add(Conv2D((3, 3), 96, LeakyRelU, padding='same'), lr_ratio=0.9)
net.add(Pool((2, 2), 2))

# Layer 4: Conv (10x10x96 -> 10x10x128)
net.add(Conv2D((3, 3), 128, LeakyRelU, padding='same'), lr_ratio=1.0)

# Flatten: 10x10x128 = 12,800 features
net.add(flatten())

# Dense layers with dropout
net.add(dense(512, LeakyRelU), lr_ratio=1.0)
net.add(Dropout(0.3))

net.add(dense(256, LeakyRelU), lr_ratio=1.0)
net.add(Dropout(0.3))

net.add(dense(128, LeakyRelU), lr_ratio=1.0)
net.add(Dropout(0.2))

# Output layer
net.add(dense(num_classes, softmax), lr_ratio=1.0)

# Compile model
net.compile()
print("✓ Model compiled successfully!")

# Display architecture
print("\nModel Architecture:")
print("-" * 70)
for i, layer in enumerate(net.layers):
    layer_name = layer.__class__.__name__
    if hasattr(layer, 'W') and layer.W.size > 1:
        w_shape = layer.W.shape
        num_params = layer.W.size + (layer.b.size if hasattr(layer, 'b') else 0)
        print(f"  Layer {i:2d}: {layer_name:15s} | Shape: {str(w_shape):30s} | Params: {num_params:,}")
    else:
        print(f"  Layer {i:2d}: {layer_name:15s} | No trainable parameters")

# Count total parameters
total_params = sum(
    layer.W.size + (layer.b.size if hasattr(layer, 'b') else 0)
    for layer in net.layers
    if hasattr(layer, 'W') and layer.W.size > 1
)
print("-" * 70)
print(f"Total Parameters: {total_params:,}")

# Check initial weight statistics
print("\nInitial Weight Statistics:")
print("-" * 70)
for i, layer in enumerate(net.layers):
    if hasattr(layer, 'W') and layer.W.size > 1:
        w_mean = float(layer.W.mean())
        w_std = float(layer.W.std())
        w_max = float(cp.abs(layer.W).max())
        print(f"  Layer {i:2d} ({layer.__class__.__name__:10s}): "
              f"mean={w_mean:7.4f}, std={w_std:.4f}, max={w_max:.4f}")

# ============================================================
# SANITY CHECK: Test forward pass
# ============================================================
print("\n" + "=" * 70)
print("SANITY CHECK: Testing forward pass")
print("=" * 70)

test_batch = cp.array(x_subset[:BATCH_SIZE])
test_labels = cp.array(y_subset[:BATCH_SIZE])

pred = net.forward(test_batch)
print(f"✓ Prediction shape: {pred.shape}")
print(f"  Prediction range: [{pred.min():.6f}, {pred.max():.6f}]")
print(f"  Prediction sum (should be ~1 per sample): {pred.sum(axis=1).mean():.6f}")

if cp.any(cp.isnan(pred)) or cp.any(cp.isinf(pred)):
    raise ValueError("ERROR: NaN or Inf in forward pass!")

loss = crossentropyloss(test_labels, pred, grad=False)
print(f"✓ Initial loss: {loss:.6f}")

# ============================================================
# TRAINING
# ============================================================
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print(f"Target: >90% validation accuracy")
print("=" * 70 + "\n")

net.train(
    loss_func=crossentropyloss,
    x=x_subset,
    y=y_subset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    decay=DECAY
)

# ============================================================
# TRAINING COMPLETE
# ============================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)

if hasattr(net, 'best_val_acc'):
    print(f"✓ Best validation accuracy: {net.best_val_acc:.4f} ({net.best_val_acc*100:.2f}%)")
    if net.best_val_acc >= 0.90:
        print("🎉 SUCCESS! Achieved >90% accuracy!")
    else:
        print(f"📊 Close! Need {(0.90 - net.best_val_acc)*100:.2f}% more for 90% target")
    
    if hasattr(net, 'best_model_filename_val'):
        print(f"✓ Best model saved as: {net.best_model_filename_val}")
else:
    print("⚠ No best validation accuracy recorded")

print("\nTraining complete! Check the saved .pkl file for the best model.")
print("=" * 70)