try:
    import cupy as cp
except ImportError:
    import numpy as cp
from PIL import Image
import os


def softmax(x, grad=False):
    """
    Softmax activation function
    
    IMPORTANT: When using softmax with cross-entropy loss, the gradient
    of the combined softmax+cross-entropy is just (y_pred - y_actual).
    This is computed in the loss function, so the softmax gradient
    should be identity (return 1) to not interfere.
    """
    if grad:
        # Return 1 (identity) because cross-entropy loss already 
        # includes the softmax derivative
        return cp.ones_like(x)
    
    # Forward pass: standard softmax
    shiftx = x - cp.max(x, axis=1, keepdims=True) 
    expx = cp.exp(shiftx)
    return expx / cp.sum(expx, axis=1, keepdims=True)


def LeakyRelU(x, grad=False):
    """LeakyReLU activation function"""
    alpha = 0.01
    if grad:
        return cp.where(x > 0, 1.0, alpha)
    return cp.maximum(alpha * x, x)


def crossentropyloss(y_actual, y_pred, grad=False):
    """
    Cross-entropy loss with softmax
    
    When grad=True, returns the gradient of the combined
    softmax + cross-entropy, which simplifies to (y_pred - y_actual)
    """
    if grad:
        # This is the derivative of softmax + cross-entropy combined
        return y_pred - y_actual
    else:
        # Clip predictions to avoid log(0)
        y_pred = cp.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -cp.mean(cp.sum(y_actual * cp.log(y_pred), axis=1))
        return loss


def augment_image(image):
    """Apply random augmentations to reduce overfitting"""
    # Random flip
    if cp.random.rand() > 0.5:
        image = image[:, ::-1, :]  # Horizontal flip
    
    # Random brightness
    brightness_factor = cp.random.uniform(0.8, 1.2)
    image = image * brightness_factor
    image = cp.clip(image, 0, 1)
    
    return image


def process_dataset(path, augment=False):
    """
    Load and process image dataset
    
    Args:
        path: Path to dataset directory
        augment: Whether to apply data augmentation
        
    Returns:
        images: List of image arrays
        labels: List of one-hot encoded labels
    """
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