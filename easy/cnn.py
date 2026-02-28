"""
reference:https://github.com/Exorust/TorchLeet/blob/main/torch/easy/cnn/CNN.ipynb
You are tasked with implementing a Convolutional Neural Network (CNN)
for image classification on the CIFAR-10 dataset using PyTorch.
The model should contain convolutional layers for feature extraction,
pooling layers for downsampling, and fully connected layers for classification.
Your goal is to complete the CNN model by defining the necessary layers and
implementing the forward pass
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(42)
# Load CIFAR-10 dataset
transform = transforms.Compose(
                [   transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
                ])
"""
transforms.normalize (mean, std, inplace=False). It is used to
scale each feature value among dataset to a z-score. This normalization
is used in image tensor since mean, std are triple tuple vectors
whose corresponds to (C,H,W) where C: number of channels, H: image height in a pixel
and WL image width in a pixel. E.g.
x = torch.empty(3, 32, 32).uniform_(0, 1)
x is an image tensor whose dimension is 3 * 32*32, i.e. X is an image of
# 224*224 pixels having 3 colors. In other words, X has color 0, 1, 2 corresponding to red, green and blue.
x[0] show the intensity of color 0 in a pixel 32*32 image.
x[1] show the intensity of color 1 in the same pixel 32*32 image.
x[2] show the intensity of color 2 in the same pixel 32*32 image.
"""

train_dataset = torchvision.datasets.CIFAR10( root='./data', train = True, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
#
test_dataset = torchvision.datasets.CIFAR10( root = './data', train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False )

"""
Reference:
https://www.tomasbeuzen.com/deep-learning-with-pytorch/chapters/chapter5_cnns-pt1.html and
https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/
channel, padding and strides in a Conv2d:
channels are features whoese each feature is a matrix.
CNN is a series of whose node is a matrix. Node in the network is known as feature/channel.
The last layer of CNN is flattened to be a vector which is an input of a fully connected graph.
In particularly, in the input layer, there are at most three nodes (red, green and blue) and each node is a matrix
which represents an intensity in 32*32 pixel.

Below is an example with:
padding=1: we have 1 layer of 0’s around our border
strides=(2,2): our kernel moves 2 data points to the right for each row, then moves 2 data points down to the next row
"""
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
         # Define a 2D convolutional layer:
        # Input channels: 3 (for RGB image)
        # Output channels: 32
        # Kernel size: 3x3
        # Padding: 1 (to maintain image size, as kernel is 3x3
        # the output layer is 32 * 32 * 32 where the first 32 is out_channels (feature),
        #                                        the 2nd 32 is the row of a matrix whose formula is #number of input pixel in a height + 2*padding_size - (kernel_size + striding - 1) + 1)
        #                                                                                         = 32 + 2 -3 + 1
        #                                        the 2nd 32 is the column of a matrix whose formula is #number of input pixel in a row + 2*padding_size - (kernel_size + striding - 1) + 1)
        #                                                                                         = 32 + 2 -3 + 1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        # Another convolutional layer:
        # Input channels: 32 (from the previous layer's output)
        # Output channels: 64
        # Kernel size: 3x3
        # the output layer should be 64 * 32 * 32 (assume the input layer is 32 * 32 * 32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        # Define a 2x2 Max Pooling layer with a stride of 2
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128) #the input later is the flatenning of last layer of the cnn network, so it is 64 * 16 * 16
        self.fc2 = nn.Linear(128, 10 ) #number of labels in the classification.
        self.relu = nn.ReLU()

    def forward(self,x):
        #Given x is 3 * 32 * 32
        y = self.relu(self.conv1(x)) #ouptut layer is 32 * 32 *32
        y = self.relu(self.conv2(y)) # output layer is 64 * 32 * 32
        y = self.pool(y) # output layer is 64 * 16 * 16
        y = y.view(y.size(0),-1) # flatten. If batch is 1, it is a vector having 64 * 16 * 16 elements,
                                 #          In this example batch is 64, so it is a 64 by (64 * 16 * 16) matrix
                                 #          y.size(0) is a batch size.
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        return y

model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#Training
epochs = 100
for epoch in range(epochs):
    for images, labels in train_loader:
        labels_pred = model(images)
        loss = criterion(labels_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss:{loss.item():.4f}")

#Evaluation on the test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        one_hot_encoding_labels = model(images)
        """
        e.g. one_hot_encoding_labels is [
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    ]
        """
        _, pred_labels = torch.max(one_hot_encoding_labels, axis = 1) #get the index of the max value in each row aka one_hot_encoding_labels.
        total += labels.size(0)
        correct += (pred_labels == labels).sum().item()
    print(f"Test Accuracy: {100 * correct/total:.4f}")

#the decrease in a loss function is a little unstable along epoch.
