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
data_path = "C:/Projects/CNN/data_resized"
images, labels = process_dataset(data_path, augment=False)

# Convert to arrays
images = cp.array(images, dtype=cp.float32)
labels = cp.array(labels, dtype=cp.float32)

# Combine into pairs
dataset = list(zip(images, labels))

# Shuffle dataset
random.shuffle(dataset)

# Optionally: take a large subset or full dataset
subset_size = 10  # full dataset
subset = dataset[:subset_size]

# Split back into images and labels
x_subset, y_subset = zip(*subset)
x_subset = cp.array(x_subset, dtype=cp.float32)
y_subset = cp.array(y_subset, dtype=cp.float32)

batch_size = 2
num_classes = y_subset.shape[1]

# -----------------------------
# MODEL SETUP
# -----------------------------
net = model()
net.add(Conv2D((5, 5), 32, LeakyRelU, padding='same', input_shape=(batch_size, 80, 80, 3), first=True))
net.add(Pool((2, 2), 2))
net.add(Conv2D((3, 3), 64, LeakyRelU, padding='same'))
net.add(Pool((2, 2), 2))
net.add(Conv2D((3, 3), 96, LeakyRelU, padding='same'))
net.add(flatten())
net.add(dense(256, LeakyRelU), lr_ratio=0.001)
net.add(Dropout(0.2))
net.add(dense(128, LeakyRelU), lr_ratio=0.001)
net.add(Dropout(0.2))
net.add(dense(num_classes, softmax))

# Compile model
net.compile()

# -----------------------------
# TRAIN
# -----------------------------
net.train(
    loss_func=crossentropyloss,
    x=x_subset,
    y=y_subset,
    epochs=15,
    batch_size=batch_size, 
    learning_rate=0.00001,
    decay=0.96
)
