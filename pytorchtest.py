import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import cupy as cp
from utils.cnn_utils import process_dataset
import random

# -----------------------------
# LOAD DATASET
# -----------------------------
data_path = "C:/Projects/CNN/data_resized"
images, labels = process_dataset(data_path, augment=False)

# Convert lists of CuPy arrays to a single NumPy array
images_np = cp.asnumpy(cp.stack(images)).astype("float32")
labels_np = cp.asnumpy(cp.stack(labels)).astype("float32")

# Shuffle dataset in unison
indices = list(range(len(images_np)))
random.shuffle(indices)
images_np = images_np[indices]
labels_np = labels_np[indices]

# Take a small subset
subset_size = 10000
images_np = images_np[:subset_size]
labels_np = labels_np[:subset_size]

# Convert to PyTorch tensors
images_tensor = torch.tensor(images_np)
labels_tensor = torch.tensor(labels_np)

# -----------------------------
# CREATE DATASET & DATALOADER
# -----------------------------
batch_size = 64
dataset = TensorDataset(images_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# DEFINE CNN
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(96 * 20 * 20, 256)  # assuming input 80x80
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        
        # Fix: make tensor contiguous before flattening
        x = x.contiguous().view(x.size(0), -1)
        
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)
        return x

num_classes = labels_tensor.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes).to(device)

# -----------------------------
# TRAINING SETUP
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# TRAIN LOOP
# -----------------------------
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # PyTorch expects class indices, not one-hot labels
        targets = torch.argmax(batch_y, dim=1)

        optimizer.zero_grad()
        outputs = model(batch_x.permute(0, 3, 1, 2))  # NHWC -> NCHW
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += batch_x.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            targets = torch.argmax(batch_y, dim=1)
            outputs = model(batch_x.permute(0, 3, 1, 2))
            loss = criterion(outputs, targets)

            val_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == targets).sum().item()
            val_total += batch_x.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
