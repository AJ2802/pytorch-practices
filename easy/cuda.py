"""
Reference:
Problem: Implement Mixed Precision Training Using torch.cuda.amp
Problem Statement
Mixed precision training uses both 16-bit and 32-bit floating-point types
to accelerate training and reduce memory usage without significantly affecting
model performance. Your task is to implement mixed precision training for a
deep learning model using PyTorch's torch.cuda.amp.
"""

# Implement mixed precision training in Pytorch using
# torch.cuda.amp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Generate synthetic data
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X,y)
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

# Initialize model, loss function and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Enable mixed precision training
# Reference: https://www.tencentcloud.com/techpedia/126165
"""
Mixed precision training in PyTorch is a technique that uses
both 16-bit (float16) and 32-bit (float32) floating-point
types to accelerate training while maintaining model accuracy.
It leverages the performance benefits of lower precision
(faster computation and reduced memory usage) while keeping
critical parts of the computation in higher precision to
avoid numerical instability.
"""
scaler = torch.cuda.amp.GradScaler()


# Training loop
epochs = 5
for epoch in range(epochs):
    for inputs, labels in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()

        """
        torch.cuda.amp.autocast(): context manager that
        automatically chooses the appropriate precision
        (float16 or float32) for each operation to maximize
        performance and numerical stability.

        Forward pass under autocast
        """
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimzer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimzer)
            scaler.update()
            # loss.backward()
            # optimzer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# test the model on the new data
X_test = torch.rand(5,10).cuda()
with torch.no_grad(), torch.cuda.amp.autocast():
    predictions = model(X_test)
    print(f"Predictions {predictions}")
