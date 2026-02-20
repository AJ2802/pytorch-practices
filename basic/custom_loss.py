#Reference: https://github.com/Exorust/TorchLeet/blob/main/torch/basic/custom-loss/custom-loss.ipynb
"""
Problem Statement
You are tasked with implementing the Huber Loss as a custom loss function
in PyTorch. The Huber loss is a robust loss function used in regression tasks,
less sensitive to outliers than Mean Squared Error (MSE). It transitions
between L2 loss (squared error) and L1 loss (absolute error) based on a
threshold parameter.

Huber loss function is as below:
                        1/2(y-y_hat)^2 if |y-y_hat|<=\delta
L_{\delta}(y, y_hat) =
                        \delta*(|y-y_hat|-1/2\delta) if |y-y_hat|>\delta

where y is the true value
y_hat is the predicted value
\delta is a threshold parameter to control the transition between L1 and L2 loss.

Exercise: check Huber loss func is continuous at |y-y_hat|=\delta.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

#Generate synthetic data
torch.manual_seed(42)
X = torch.rand(100,1)*10
y = 2*X + 3 + torch.rand(100,1)

class HuberLoss(nn.Module):
    def __init__(self,delta = 1.0):
        super().__init__()
        self.delta = delta

    """
    Note we return mean of the total loss in the end.
    It is okay if we return sum of the total loss instead.
    In this case, we need to make epochs larger by 10 and
    learning rate scaled down by 100, i.e.
    epochs : 1000 -> 10000
    lr : 0.01 --> 0.0001
    It is because the sum of the total loss is 100 times larger
    than mean of the total loss in this example.
    If we does not do any scaling of these parameters and keep
    them same as mean of the total loss situation, i.e.
        epochs : 1000
        lr : 0.01
    the loss function do not converge.
    """
    def forward(self,y_pred, y_truth):
        loss_tensor = torch.where(
            (y_pred-y_truth) < self.delta,
            1/2*torch.pow((y_pred-y_truth), 2),
            self.delta * (torch.abs(y_pred - y_truth) - 1/2*self.delta)
            )
        return torch.mean(loss_tensor)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

# Initialize model, loss function, and optimizer
model = LinearRegressionModel()
criterion = HuberLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    y_pred = model.forward(X)
    loss = criterion.forward(y_pred, y)

    #Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1 }/{epochs}], Loss: {loss.item():.4f}")

# Display the learned paramerers
[w, b] = model.linear.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Plot the model fit to the train data
plt.figure(figsize=(4,4))
plt.scatter(X, y, label = 'Training Data')
plt.plot(X, w.item()*X+b.item(), label = "Model Fit")
plt.legend()
plt.show()

#Testing on new data
X_test = torch.tensor([[4.0], [7.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
