"""
Reference: https://github.com/Exorust/TorchLeet/blob/main/torch/medium/cnn-scratch/CNN_scratch_SOLN.ipynb
Build cnn from scratch instead of using
Conv2d and MaxPool2d built in function
in pytorch library. It can help us to understand the mathematics inside cnn neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(42)
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
            ])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train = True, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)

test_dataset = torchvision.datasets.CIFAR10(root="./data", train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)

class Conv2dCustom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(Conv2dCustom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn((self.out_channels, *self.kernel_size))*0.1)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x):
        batch_size, in_channels, h, w = x.shape
        kh, kw = self.kernel_size
        sh = sw = self.stride
        ph = pw = self.padding

        x_padded = F.pad(x, (pw, pw, ph, ph)) #up, down, left, right

        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1

        out = torch.zeros((batch_size, self.out_channels, oh, ow), device = x.device)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(oh):
                    for j in range(ow):
                        h_start = i*sh
                        h_end = i*sh + kh

                        w_start = j*sw
                        w_end = j*sw + kw

                        region = x_padded[b,:, h_start:h_end, w_start: w_end]
                        out[b, oc, i, j] = torch.sum(self.weight[oc] * region) + self.bias[oc]
        return out

class MaxPool2dCustom(nn.Module):
    def __init__(self, kernel_size, stride = None):
        super(MaxPool2dCustom, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride != None else kernel_size

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        kh, kw = self.kernel_size
        sh = sw = self.stride

        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1

        out = torch.zeros((batch_size, channels, oh, ow))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(oh):
                    for j in range(ow):
                        h_start = i * sh
                        h_end = i * sh + kh

                        w_start = j * sw
                        w_end = j * sw + kw

                        region = x[b,c, h_start: h_end, w_start:w_end]
                        out[b,c,i,j] = torch.max(region)
        return out

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dCustom(3, 8, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = Conv2dCustom(8, 16, kernel_size = 3, stride = 1, padding = 1)
        self.pool = MaxPool2dCustom(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(16*16*16,64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 10
for epoch in range(epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim = 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")
