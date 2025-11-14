import numpy as cp
from model.model import model
from model.dense import dense
from model.Conv2D import Conv2D
from model.flatten import flatten
from model.pool import Pool
from utils.cnn_utils import LeakyRelU, softmax, crossentropyloss, process_dataset

# Dummy input (batch_size=2, features=10)
x = cp.random.randn(32, 240, 320, 3)

# Dummy labels (e.g., for classification into 5 classes)
y_true_small = cp.array([[0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0]])
y_true = cp.tile(y_true_small, (16, 1))


# ---- MODEL SETUP ----
net = model()
net.add(Conv2D((3, 3), 64, LeakyRelU, padding='same', input_shape=(32, 240, 320, 3), first=True))
net.add(Pool((5, 5), 5))
net.add(Conv2D((3, 3), 64, LeakyRelU, padding='same'))
net.add(Pool((5, 5), 5))
net.add(Conv2D((5, 5), 96, LeakyRelU, padding='same'))
net.add(flatten())
net.add(dense(512, LeakyRelU), lr_ratio=0.001)
net.add(dense(512, LeakyRelU), lr_ratio=0.001)
net.add(dense(5, softmax), lr_ratio=1.0)

# Compile model with input shape
net.compile()

# ---- FORWARD PASS ----
y_pred = net.forward(x)
print("Predictions:\n", y_pred)

# ---- BACKWARD PASS ----
grad = crossentropyloss(y_true, y_pred, grad=True)
net.backward(grad)

# --- Save weights before update ---
W_before = []
for i, layer in enumerate(net.layers):
    if hasattr(layer, 'W'):
        W_before.append(layer.W.copy())
        print(f"Saved weights for layer {i}")
    else:
        W_before.append(None)
        print(f"No weights found in layer {i}")

# ---- ADAM OPTIMIZER UPDATE ----
net.update(0.001, 2)

# --- Save weights after update ---
W_after = []
for i, layer in enumerate(net.layers):
    if hasattr(layer, 'W'):
        W_after.append(layer.W)
    else:
        W_after.append(None)

# ---- CHECK IF WEIGHTS UPDATED ----
for i, (w_before, w_after) in enumerate(zip(W_before, W_after)):
    if w_before is None or w_after is None:
        print(f"Layer {i}: No weights to compare.")
    else:
        updated = not cp.allclose(w_before, w_after)
        print(f"Layer {i} weights updated: {updated}")
        
# ---- CHECK LOSS ----
loss = crossentropyloss(y_true, y_pred)
print("Loss:", float(loss))

x, y = process_dataset("C:/Users/ey2ma/Downloads/Avatar.ai/data")

net.train(crossentropyloss, x, y)


