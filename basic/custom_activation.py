# Problem Statement
# You are tasked with implementing a custom activation function in PyTorch that computes the following operation:
# tanh(x) + x
# Once implemented, this custom activation function will be used in a simple linear regression model.
#
# Reference:
# https://github.com/Exorust/TorchLeet/blob/main/torch/basic/custom-activation/custom-activation_SOLN.ipynb
# https://stackoverflow.com/questions/55765234/pytorch-custom-activation-functions
# https://discuss.pytorch.org/t/how-to-define-a-parametrized-tanh-activation-function-with-2-trainable-parameters/161720


import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

#Gernate synthetic data
torch.manual_seed(42)
X = torch.rand(100, 1)* 10
y = 2 * X + 3 + torch.rand(100,1)

# Define the Linear Regression Model
class CustomActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1) # Single input and single output

    #forward pass
    def custom_activation(self,x):
        return x + torch.tanh(x)

    def forward(self, x):
        return self.custom_activation(self.linear(x))

#Initialize the model, loss function, and optimizer
model = CustomActivationModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 1000
for epoch in range(epochs):
    #Forward pass
    prediction = model.forward(X)
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1 ) % 100 == 0:
        print(f'Epoch [{epoch + 1 }/{epochs}], Loss: {loss.item():.4f}')


#Display the learned parameters
[w,b] = model.linear.parameters()
print(f'Linear weight: {w.item():.4f}, Learned bias: {b.item():.4f}')

# Plot the model fit to the train data
plt.figure(figsize=(4,4))
plt.scatter(X, y, label = 'Training Data')
plt.plot(X, w.item()*X+b.item(), 'r', label = 'Model Fit')
plt.legend()
plt.show()

#Testing on new data
X_test = torch.tensor([[4.0], [7.0]])
with torch.no_grad():
    prediction = model(X_test)
    print(f'Predictions for {X_test.tolist()}: {prediction.tolist()}')
