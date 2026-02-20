# Problem: Implement Linear Regression
# Problem Statement
# Your task is to implement a Linear Regression model using PyTorch. The model should predict a continuous target variable based on a given set of input features.
#
# Requirements
# Model Definition:
# Implement a class LinearRegressionModel with:
# A single linear layer mapping input features to the target variable.
# Forward Method:
# Implement the forward method to compute predictions given input data.
# Reference:
# https://github.com/Exorust/TorchLeet/blob/main/torch/basic/lin-regression/lin-regression.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
torch.manual_seed(42)
X = torch.rand(100,1)*10 #100 data points between 0 and 10 . Each data has only one value
y = 2 *X +3 + torch.rand(100,1) #linear relationship with noise

# Define the Linear Regression Model
#TODO: Add the layer and forward implementation


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)
        # Example: A linear layer torch.nn.Linear(x,y) that takes x input features and outputs y features

    def forward(self, x):
        # Forward pass: apply the linear transformation
        y_pred = self.linear(x)
        return y_pred

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error is standard for regression
optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent optimizer



#Training:
epochs = 1000
for epoch in range(epochs):
    #Forward pass
    predictions = model(X)

    loss = criterion(predictions, y)

    #Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Log progress every 100 epochs
    if (epoch + 1 )%100 == 0:
        print(f'Epoch [{epochs + 1}/{epochs}], Loss {loss.item():.4f}')

#Plot the results:
plt.plot(X, y, 'go', label = 'True data', alpha = 0.5)
plt.plot(X, predictions.detach().numpy(), '--', label = 'Predictions', alpha= 0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


print(f'model info: {model}')
# for name, param in model.named_parameters():
#     print(f'Layer Name: {name} and tensor parameters values: {param.data}')
#     if param.grad is not None:
#         print(f'Gradient for {name} is {param.grad}')
#     else:
#         print(f'Gradient for {name} is {math.nan}')

#Display the learned parameters
[w, b] = model.linear.parameters()
print(f'Learned weight: {w.item()}:4,f, Learned bias: {b.item():.4f}')

# Testing on new data
X_test = torch.tensor([[4.0], [7.0]])
"""
Using with torch.no_grad() disables gradient calculation.
So, the reason why it uses less memory is that it’s not storing
any Tensors that are needed to calculate gradients of your loss.
Also, because you don’t store anything for the backward pass,
the evaluation of your network is quicker (and use less memory).
"""
with torch.no_grad():
    predictions = model(X_test)
    print(f'Predictions for {X_test.tolist()}: {predictions.tolist()}')
