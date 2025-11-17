try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

print("Backend:", "CuPy (GPU)" if GPU_AVAILABLE else "NumPy (CPU)")

import random
from model.model import model
from model.dense import dense
from model.Conv2D import Conv2D
from model.flatten import flatten
from model.pool import Pool
from model.dropout import Dropout
from utils.cnn_utils import LeakyRelU, softmax, crossentropyloss, process_dataset

# -----------------------------
# LOAD DATASET
# -----------------------------
print("Loading dataset...")
data_path = "C:/Projects/CNN/data_resized"
images, labels = process_dataset(data_path, augment=False)

# Convert to arrays
images = cp.array(images, dtype=cp.float32)
labels = cp.array(labels, dtype=cp.float32)

print(f"Dataset loaded: {len(images)} images")
print(f"Image shape: {images[0].shape}")
print(f"Data stats - Mean: {images.mean():.4f}, Std: {images.std():.4f}")

# Combine into pairs
dataset = list(zip(images, labels))

# Shuffle dataset
random.shuffle(dataset)

# Use full dataset or subset for faster experiments
USE_FULL_DATASET = False  # Set to True when ready
if USE_FULL_DATASET:
    subset = dataset
    print("Using FULL dataset")
else:
    subset_size = 2000
    subset = dataset[:subset_size]
    print(f"Using subset: {subset_size} images")

# Split back into images and labels
x_subset, y_subset = zip(*subset)
x_subset = list(x_subset)
y_subset = list(y_subset)

batch_size = 64
num_classes = y_subset[0].shape[0]

print(f"Batch size: {batch_size}")
print(f"Number of classes: {num_classes}")

# -----------------------------
# MODEL SETUP
# -----------------------------
print("\nBuilding model...")
net = model()
net.add(Conv2D((5, 5), 32, LeakyRelU, padding='same', input_shape=(batch_size, 80, 80, 3), first=True), lr_ratio=1.0)
net.add(Pool((2, 2), 2))
net.add(Conv2D((3, 3), 64, LeakyRelU, padding='same'), lr_ratio=1.0)
net.add(Pool((2, 2), 2))
net.add(Conv2D((3, 3), 96, LeakyRelU, padding='same'), lr_ratio=1.0)
net.add(flatten())
net.add(dense(256, LeakyRelU), lr_ratio=1.0)
net.add(Dropout(0.2))
net.add(dense(128, LeakyRelU), lr_ratio=1.0)
net.add(Dropout(0.2))
net.add(dense(num_classes, softmax), lr_ratio=1.0)

# Compile model
net.compile()

print("✓ Model compiled successfully!")

# -----------------------------
# TRAIN
# -----------------------------
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

net.train(
    loss_func=crossentropyloss,
    x=x_subset,
    y=y_subset,
    epochs=25,
    batch_size=batch_size, 
    learning_rate=0.00005,  # Good starting point
    decay=0.97
)

print("\n✓ Training complete!")
print(f"Best validation accuracy: {net.best_val_acc:.4f}")