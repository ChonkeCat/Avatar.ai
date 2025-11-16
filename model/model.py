#model architecture definition, e.g. forward pass, etc.
try:
    import cupy as cp
except ImportError:
    import numpy as cp
import pickle
import random
import os

class model():
    def __init__(self):
        self.layers = []
        self.learning_rate_mask = []
        # add stats later

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
        return gradient
    
    def update(self, learning_rate, beta1=0.9, beta2=0.99):
        for layer, lr_ratio in zip(self.layers, self.learning_rate_mask):
            actual_learning_rate = lr_ratio * learning_rate

            grad_W = layer.dW
            grad_b = layer.db

            # Adam updates for weights
            layer.mo = beta1 * layer.mo + (1 - beta1) * grad_W
            layer.acc = beta2 * layer.acc + (1 - beta2) * (grad_W * grad_W)
            layer.W -= actual_learning_rate * layer.mo / (cp.sqrt(layer.acc) + 1e-7)

            # Adam updates for biases
            layer.mo_b = beta1 * layer.mo_b + (1 - beta1) * grad_b
            layer.acc_b = beta2 * layer.acc_b + (1 - beta2) * (grad_b * grad_b)
            layer.b -= actual_learning_rate * layer.mo_b / (cp.sqrt(layer.acc_b) + 1e-7)

    def train(self, loss_func, x, y, epochs = 50, learning_rate = 0.001, decay = 0.96, batch_size = 32):
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
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_num, (batch_x, batch_y) in enumerate(train_batches, start=1):
                pred = self.forward(batch_x)
                total_loss += loss_func(batch_y, pred, grad=False)

                self.backward(loss_func(batch_y, pred, grad=True))

                self.update(learning_rate)

                total_correct += one_hot_accuracy(pred, batch_y)
                total_samples += batch_x.shape[0]

                print(f"Processed train batch {batch_num}/{len(train_batches)}")
                    
            epoch_loss = total_loss / total_samples
            epoch_acc  = total_correct / total_samples

            val_loss = 0
            val_correct = 0
            val_samples = 0

            for batch_num, (batch_x, batch_y) in enumerate(val_batches, start=1):
                pred = self.forward(batch_x)
                
                val_loss += loss_func(batch_y, pred, grad=False)
                val_correct += one_hot_accuracy(pred, batch_y)
                val_samples += batch_x.shape[0]

                print(f"Processed val batch {batch_num}/{len(val_batches)}")

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
                print("Saving...")
                self.save(self.best_model_filename_val)

            learning_rate *= decay

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            
    def save(self, path):
        save = open(path, "wb")
        pickle.dump(self, save)
        save.close()

    def load(self, path):
        load = open(path, "rb")
        model = pickle.load(load)
        load.close()
        return model