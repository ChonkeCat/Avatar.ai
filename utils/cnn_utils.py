#helper functions for the CNN, e.g. loading dataset, ReLU, padding, etc.
try:
    import cupy as cp
except ImportError:
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
    alpha = 0.01
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

def augment_image(image):
    """Apply random augmentations to reduce overfitting"""
    # Random flip
    if cp.random.rand() > 0.5:
        image = image[:, ::-1, :]  # Horizontal flip
    
    # Random rotation (small angles)
    if cp.random.rand() > 0.5:
        angle = cp.random.uniform(-10, 10)
        # Implement rotation logic here
    
    # Random brightness
    brightness_factor = cp.random.uniform(0.8, 1.2)
    image = image * brightness_factor
    image = cp.clip(image, 0, 1)
    
    return image

def process_dataset(path, augment=False):
    images = []
    labels = []
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    for i, folder in enumerate(folders):
        label = cp.zeros(len(folders), dtype=cp.float32)
        label[i] = 1.0

        folder_path = os.path.join(path, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                try:
                    img = Image.open(file_path).convert("RGB")
                    img_array = cp.asarray(img).astype(cp.float32) / 255.0
                    images.append(img_array)
                    labels.append(label)
                    
                    # Add augmented versions
                    if augment:
                        augmented_img = augment_image(img_array)
                        images.append(augmented_img)
                        labels.append(label)
                        
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

    return images, labels
