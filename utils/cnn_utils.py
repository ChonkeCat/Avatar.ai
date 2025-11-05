#helper functions for the CNN, e.g. loading dataset, ReLU, padding, etc.
import numpy as cp
from PIL import Image
import os


## Computes the softmax for a 1-D vector
## inputs: x -> 1-D vector, grad -> true if you want the gradiant, false otherwise
## outputs: 1-D vector corresponding to the normalized verison on the input
##          or nxn jacobian of the softmax

def softmax(x, grad = False):
    if grad:
        return x
    shiftx = x - cp.max(x, axis=1, keepdims=True) 
    expx = cp.exp(shiftx)
    return expx / cp.sum(expx, axis=1, keepdims=True)


## LeakyRelU activation function, returns x if it is greater than or equal to 0, returns 0 otherwise
## inputs: x -> float, grad => true if you want the gradiant, false otherwise

def LeakyRelU(x, grad=False):
    alpha = 0.05
    if grad:
        return cp.where(x > 0, 1.0, alpha)
    return cp.maximum(alpha * x, x)

def crossentropyloss(y_actual, y_pred, grad=False):
    if grad:
        return y_pred - y_actual
    else:
        y_pred = cp.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -cp.mean(cp.sum(y_actual * cp.log(y_pred), axis=1))
        return loss

def process_dataset(path):
    images = []
    labels = []
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    for i, folder in enumerate(folders):
        label = [0] * len(folders)
        label[i] = 1

        folder_path = os.path.join(path, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                try:
                    img = Image.open(file_path).convert("RGB")
                    images.append(cp.asarray(img))
                    labels.append(label)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

    return images, labels
