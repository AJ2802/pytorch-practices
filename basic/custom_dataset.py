# Reference: https://github.com/Exorust/TorchLeet/blob/main/torch/basic/custom-dataset/custom-dataset.ipynb

import torch
import pandas as pd

torch.manual_seed(42)
X = torch.rand(100, 1) * 10 # 100 data points between 0 and 10
y = 2 * X + 3 + torch.rand(100,1) # Linear relationship with noise

# Save the generated data to data.csv
data = torch.cat((X,y), dim = 1)
#dim if dim = 0, it means concat y to X by adding rows, in other word, y and x must have the same number of columns
# i.e. if X is 2 by 3 matrix, and y is 4 by 3 matric, then cat((X,y), dim=0) is 6 by 3 matrix
#dim if dim = 1, it means concat y to X by adding columns, in other word, y and x must have the same number of rows
# i.e. if X is 2 by 3 matrix, and y is 2 by 4 matric, then cat((X,y), dim=1) is 2 by 7 matrix

df = pd.DataFrame(data.numpy(), columns = ['X', 'y'])
df.to_csv('data.csv', index=False)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

class LinearRegressionDataset(Dataset):
    def __init__(self, data_csv):
        self.dataset = pd.read_csv(data_csv)
        #check def view explanation on https://stackoverflow.com/questions/42479902/what-does-view-do-in-pytorch
        self.X = torch.tensor(self.dataset['X'].values, dtype = torch.float32).view(-1,1) # view is to reshape it from one dim vector 100 to a 2 dimension matrix 100 X 1
        self.y = torch.tensor(self.dataset['y'].values, dtype = torch.float32).view(-1,1) # view is to reshape it from one dim vector 100 to a 2 dimension matrix 100 X 1


    def __len__(self): #it is used in LinearRegressionDataset('data.csv') below
        """
        Returns the total number of samples in the datasets.
        """
        # This is the essential part: returning the length of the loaded data structure
        return len(self.dataset)

    def __getitem__(self, idx): #it is used in for batch_X, batch_y in dataloader below
        return self.X, self.y

# Example usage of the DataLoader
dataset = LinearRegressionDataset('data.csv')

dataloader = DataLoader( dataset = dataset,  # Your dataset object (mandatory)
                         batch_size = 10,  # Number of samples per batch
                         shuffle = True, # Shuffle data every epoch (useful for training)
                         num_workers = 0,  # Number of subprocesses for data loading (for parallel fetching)
                         pin_memory = False)  # Speeds up CPU-to-GPU data transfer

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x) # Single input and single output
        return y_pred

# Initialize the model, loss function and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr =0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        #Forward pass
        y_pred = model.forward(batch_X)
        loss = criterion(y_pred, batch_y)

        # Backward pass and optimization
        # clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
        optimizer.zero_grad()

        # The loss function is in term of parameters in a model. loss.backward calculate the gradient of a loss function (see https://www.geeksforgeeks.org/python/python-pytorch-backward-function/) Without this func, gradiation is none.
        loss.backward() #backward propagation

        #makes the optimizer iterate over all parameters (tensors) it is supposed to update and use their internally stored grad to update their values.
        optimizer.step() # update parameters.

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Display the learned parameters
[w, b] = model.linear.parameters()
print(f'Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}')

# Testing on new data
X_test = torch.tensor([[4.0], [7.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f'Predictions for {X_test.tolist()}: {predictions.tolist()}')
