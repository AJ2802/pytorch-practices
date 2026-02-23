"""
Reference: https://github.com/Exorust/TorchLeet/blob/main/torch/basic/custom-DNN/custon-DNN.ipynb
You are tasked with constructing a Deep Neural Network (DNN) model to
solve a regression task using PyTorch. The objective is to predict
target values from synthetic data exhibiting a non-linear relationship.
"""

import torch
import torch.nn as nn
import torch.optim as optim

#Generate synthetic data
torch.manual_seed(42)
X = torch.rand(100,2)*10 # 100 data points with 2 features
y = (X[:,0] + X[:,1]*2).unsqueeze(1) + torch.rand(100, 1) #Non-linear relationship
"""
unsqueeze is to one more dimension in 2nd position.
In an example of a tensor [[1,2],[3,4]] whose dimension is 2 * 2
For unsqueeze(0),
the above tensor transform to a tensor dimension 1 * 2 * 2 as below
                                    [
                                        [[1,2],[3,4]]
                                    ]
For unsqueeze(1),
the above tensor transform to a tensor dimension 2 * 1 * 2 as below
                                    [
                                            [[1,2]],
                                            [[3,4]]
                                        ]
For unsqueeze(2),
the above tensor transform to a tensor dimension 2 * 2 * 1 as below
                                    [
                                        [[1],[2]],
                                        [[3],[4]]
                                        ]
Note: The additional bracket from the most outer to the
most inner when the value increases from 0 to 2.
More refererenc is find on
https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
"""

# Define the Deep Neural Model
class DNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,10)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(10,1)
        """ If only one layer and one activation func,
        the model is too simple and
        it cannot be trainined to have the
        above non-linear relation in y.
        """
    #forward pass
    def forward(self,x):
        return self.layer2(self.activation(self.layer1(x)))

model = DNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

#Testing on new data
X_test = torch.tensor([[4.0,3.0], [7.0,8.0]])
"""
Note input this tensor torch.tensor([[4,3], [7,8]]) to
the DNN model give a syntax error coz the model only allow
float type input values.
"""
with torch.no_grad():
    y_pred = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {y_pred.tolist()}")
