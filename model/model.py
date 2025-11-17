#model architecture definition, e.g. forward pass, etc.
try:
    import cupy as cp
except ImportError:
    import numpy as cp
import pickle
import random
import os
from tqdm import tqdm

class model():
    def __init__(self):
        self.layers = []
        self.learning_rate_mask = []
        self.t = 0  # Time step for Adam bias correction

    def add(self, layer, lr_ratio=1):
        self.layers.append(layer)
        self.learning_rate_mask.append(lr_ratio)

    def insert(self, layer, lr_ratio, i):
        self.layers.insert(i, layer)
        self.learning_rate_mask.append(lr_ratio)

    def pop(self):
        self.layers = self.layers[:-1]

    def compile(self):
        output_shape = self.layers[0].initialize(self.layers[0].input_shape)
        for i in range(len(self.layers[1:])):
            output_shape = self.layers[i + 1].initialize(output_shape)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            # Check for NaN/Inf in gradients
            if cp.any(cp.isnan(gradient)) or cp.any(cp.isinf(gradient)):
                print(f"WARNING: NaN/Inf detected in gradient from {layer.__class__.__name__}")
        return gradient
    
    def clip_gradients(self, max_norm=1.0):
        """Clip gradients by global norm to prevent explosion"""
        total_norm = 0.0
        for layer in self.layers:
            if hasattr(layer, 'dW') and layer.dW.size > 1:
                total_norm += cp.sum(layer.dW ** 2)
            if hasattr(layer, 'db') and layer.db.size > 1:
                total_norm += cp.sum(layer.db ** 2)
        
        total_norm = cp.sqrt(total_norm)
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for layer in self.layers:
                if hasattr(layer, 'dW') and layer.dW.size > 1:
                    layer.dW *= clip_coef
                if hasattr(layer, 'db') and layer.db.size > 1:
                    layer.db *= clip_coef
        
        return float(total_norm)
    
    def update(self, learning_rate, beta1=0.9, beta2=0.999, batch_size=32):
        self.t += 1
        
        # FIRST: Divide gradients by batch size
        for layer in self.layers:
            if hasattr(layer, 'dW') and layer.dW.size > 1:
                layer.dW = layer.dW / batch_size  # Don't use /=, create new array
            if hasattr(layer, 'db') and layer.db.size > 1:
                layer.db = layer.db / batch_size
        
        # SECOND: Clip gradients BEFORE computing norms for display
        total_norm_sq = 0.0
        for layer in self.layers:
            if hasattr(layer, 'dW') and layer.dW.size > 1:
                total_norm_sq += float(cp.sum(layer.dW ** 2))
            if hasattr(layer, 'db') and layer.db.size > 1:
                total_norm_sq += float(cp.sum(layer.db ** 2))
        
        total_norm = cp.sqrt(total_norm_sq)
        
        # Clip to prevent explosion
        max_norm = 5.0
        clip_coef = min(max_norm / (float(total_norm) + 1e-8), 1.0)
        
        if clip_coef < 1.0:
            for layer in self.layers:
                if hasattr(layer, 'dW') and layer.dW.size > 1:
                    layer.dW = layer.dW * clip_coef
                if hasattr(layer, 'db') and layer.db.size > 1:
                    layer.db = layer.db * clip_coef
        
        # THIRD: Apply Adam updates
        for lr_idx, layer in enumerate(self.layers):
            lr_ratio = self.learning_rate_mask[lr_idx]
            actual_lr = lr_ratio * learning_rate
            
            if hasattr(layer, 'dW') and layer.dW.size > 1:
                # Adam for weights
                layer.mo = beta1 * layer.mo + (1 - beta1) * layer.dW
                layer.acc = beta2 * layer.acc + (1 - beta2) * (layer.dW ** 2)
                
                mo_corrected = layer.mo / (1 - beta1 ** self.t)
                acc_corrected = layer.acc / (1 - beta2 ** self.t)
                
                layer.W = layer.W - actual_lr * mo_corrected / (cp.sqrt(acc_corrected) + 1e-8)
                
            if hasattr(layer, 'db') and layer.db.size > 1:
                # Adam for biases
                layer.mo_b = beta1 * layer.mo_b + (1 - beta1) * layer.db
                layer.acc_b = beta2 * layer.acc_b + (1 - beta2) * (layer.db ** 2)
                
                mo_b_corrected = layer.mo_b / (1 - beta1 ** self.t)
                acc_b_corrected = layer.acc_b / (1 - beta2 ** self.t)
                
                layer.b = layer.b - actual_lr * mo_b_corrected / (cp.sqrt(acc_b_corrected) + 1e-8)
        
        return float(total_norm)  # Return BEFORE clipping for monitoring

    def train(self, loss_func, x, y, epochs = 50, learning_rate = 0.001, decay = 0.96, batch_size = 64):
        combined = list(zip(x, y))
        random.shuffle(combined)
        x, y = zip(*combined)
        x = list(x)
        y = list(y)

        split_point = int(len(x) * 0.8)
        train_x, val_x = x[:split_point], x[split_point:]
        train_y, val_y = y[:split_point], y[split_point:]

        def batchify(a, b, batch_size):
            for i in range(0, len(a), batch_size):
                yield cp.array(a[i:i+batch_size]), cp.array(b[i:i+batch_size])

        train_batches = [(bx, by) for bx, by in batchify(train_x, train_y, batch_size)
            if bx.shape[0] == batch_size]

        val_batches = [(bx, by) for bx, by in batchify(val_x, val_y, batch_size)
            if bx.shape[0] == batch_size]

        def one_hot_accuracy(y_pred, y_true):
            pred_class = cp.argmax(y_pred, axis=-1)
            true_class = cp.argmax(y_true, axis=-1)
            return (pred_class == true_class).sum()

        for epoch in range(epochs):
            with tqdm(total=len(train_batches), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                total_loss = 0
                total_correct = 0
                total_samples = 0

                for batch_num, (batch_x, batch_y) in enumerate(train_batches, start=1):
                    # Zero gradients before backward pass
                    for layer in self.layers:
                        if hasattr(layer, 'dW') and layer.dW.size > 1:
                            layer.dW[:] = 0
                        if hasattr(layer, 'db') and layer.db.size > 1:
                            layer.db[:] = 0
                    
                    pred = self.forward(batch_x)
                    
                    # Check for NaN in predictions
                    if cp.any(cp.isnan(pred)) or cp.any(cp.isinf(pred)):
                        print(f"\nWARNING: NaN/Inf in predictions at batch {batch_num}")
                        print(f"Pred min: {pred.min()}, max: {pred.max()}")
                        break
                    
                    loss = loss_func(batch_y, pred, grad=False)
                    grad = loss_func(batch_y, pred, grad=True)
                    
                    # CRITICAL FIX: Normalize gradient from loss function by batch size FIRST
                    # This prevents overflow before backward pass
                    grad = grad / batch_x.shape[0]
                    
                    total_loss += loss * batch_x.shape[0]
                    
                    self.backward(grad)
                    
                    # Now gradients are already averaged, so pass batch_size=1 to update
                    # to prevent double division
                    grad_norm = self.clip_gradients(max_norm=5.0)
                    self.update(learning_rate, batch_size=1)  # batch_size=1 since already normalized
                    
                    total_correct += one_hot_accuracy(pred, batch_y)
                    total_samples += batch_x.shape[0]
                    
                    pbar.set_postfix({
                        'Loss': f"{(total_loss/total_samples):.4f}",
                        'Acc': f"{(total_correct/total_samples):.4f}",
                        'GradNorm': f"{grad_norm:.2f}"
                    })
                    pbar.update(1)
                        
                epoch_loss = total_loss / total_samples
                epoch_acc  = total_correct / total_samples

                val_loss = 0
                val_correct = 0
                val_samples = 0

                for batch_num, (batch_x, batch_y) in enumerate(val_batches, start=1):
                    pred = self.forward(batch_x)
                    
                    val_loss += loss_func(batch_y, pred, grad=False) * batch_x.shape[0]
                    val_correct += one_hot_accuracy(pred, batch_y)
                    val_samples += batch_x.shape[0]


                val_loss = val_loss / val_samples
                val_acc = val_correct / val_samples

                if not hasattr(self, "best_val_acc"):
                    self.best_val_acc = 0
                    self.best_model_filename_val = None

                if val_acc > self.best_val_acc:
                    if self.best_model_filename_val is not None:
                        if os.path.exists(self.best_model_filename_val):
                            os.remove(self.best_model_filename_val)

                    self.best_val_acc = val_acc
                    self.best_model_filename_val = f"val_acc{val_acc:.4f}_vloss{val_loss:.4f}_epoch{epoch+1}.pkl"
                    print("\nSaving...")
                    self.save(self.best_model_filename_val)

                learning_rate = learning_rate * decay

                print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
    def save(self, path):
        save = open(path, "wb")
        pickle.dump(self, save)
        save.close()

    def load(self, path):
        load = open(path, "rb")
        model = pickle.load(load)
        load.close()
        return model