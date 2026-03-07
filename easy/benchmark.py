"""
Reference: https://github.com/Exorust/TorchLeet/blob/main/torch/easy/benchmark/bench.ipynb

You are tasked with implementing a simple neural network model with fully
connected layers and adding benchmarking functionality to measure and
display the time taken for each epoch of training and testing.
The goal is to evaluate the model's performance and record the time
taken for both training and testing phases.

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Load MNIST dataset
torch.manual_seed(42)
transform = transforms.Compose(
            [ transforms.ToTensor(),
              transforms.Normalize((0.5,), (0.5,))
            ])

train_dataset = torchvision.datasets.MNIST( root='./root',
                                             train = True,
                                             download = True,
                                             transform = transform
                                            )
train_loader = torch.utils.data.DataLoader( train_dataset,
                                            batch_size = 64,
                                            shuffle = True
                                          )

test_dataset = torchvision.datasets.MNIST(
                                            root = './data',
                                            train = False,
                                            download = True,
                                            transform = transform,
                                         )
test_loader = torch.utils.data.DataLoader( test_dataset,
                                            batch_size = 64,
                                            shuffle = False
                                         )

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        #Note image.shape originally is 64*1*28*28 (0th dim x 1st dim x 2nd dim x 3rd dim)
        #flattening starts from 1st dim to 3rd dim.
        y = x.flatten(start_dim = 1,end_dim = 3)
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        y = self.fc4(y)
        return y

#Initlization
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

#Training
epochs = 5
for epoch in range(epochs):
    start_time = time.time()
    for images , labels in train_loader:

        pred = model(images)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time: {training_time:.4f}s")

# Evaluatio the model on the test set and benchmark the accuarcy
correct = 0
total = 0
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        pred_one_hot = model(images)
        """
        e.g. one_hot_encoding_labels is [
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    ]
        _, pred_labels = torch.max(pred_one_hot, axis = 1)
        pred_labels is the index of the max value in a row.
        e.g. the first one of pred_labels is 1
            the second one of pred_labels is 8.
        """
        _, pred_labels = torch.max(pred_one_hot, axis = 1)
        """
        pred_labels == labels is e.g. =
        tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True, False,  True,  True, False,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True, False,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        False,  True, False,  True])
        so (pred_labels == labels).shape is 64
        and (pred_labels == labels).sum()=(pred_labels == labels).sum(dim=0) # sum all True values along the 0th dim.
        """
        correct += (pred_labels == labels).sum().item()
        total += labels.size(0)
testing_time = time.time() - start_time
accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.4f}%. Testing time = {testing_time:.4f}s")
