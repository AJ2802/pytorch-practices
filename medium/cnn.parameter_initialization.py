"""
reference: https://github.com/Exorust/TorchLeet/blob/main/torch/medium/cnn-param-init/CNN_ParamInit.ipynb
Problem Statement
You are tasked with employing and evaluating a CNN model's parameter initialization strategies in Pytorch. Your goal is to initialize the weights and biases of a vanilla CNN model provided in the problem statement and comment on the implications of each strategy.

Requirements
Initialize weights and biases in the following ways:
Zero Initialization: set the parameters to zero
Random Initialization: sets model parameters to random values drawn from a normal distribution
Xavier Initialization sets them to random values from a normal distribution with mean=0 and variance=1/n
Kaiming He Initialization initializes to random values from a normal distribution with mean=0 and variance=2/n
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(42)
transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])

train_dataset = torchvision.datasets.CIFAR10(
                root = "./data",
                train = True,
                download = True,
                transform = transform)
train_loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle = True,
                batch_size = 32)

test_dataset = torchvision.datasets.CIFAR10(
                root = "./data",
                train = False,
                download = True,
                transform = transform)
test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size = 32,
                shuffle = False)

def train_test_loop(model, train_loader, test_loader, epochs = 10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
            pred = model(images)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print(f"Training loss at epoch {epoch} = {loss.item()}")

    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        pred_test = model(images)
        _, pred_test_index = torch.max(pred_test, dim = 1)
        correct += (pred_test_index==labels).sum().item() #sum(dim = 0)
        total += labels.size(0)
    print(f"Test Accuracy = {(correct * 100)/total}")


class VanillaCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1) #32*32 pixel --> (32+2-3+1)*(32+2-3+1) = (32*32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2) # 32*32 pixel --> 16*16 pixel
        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x=x.view(x.size(0),-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def config_init(init_type = "kaiming"):
    """
    nn.kaming_normal_ parameters
    mode (Literal['fan_in', 'fan_out'])
        – either 'fan_in' (default) or 'fan_out'.
        Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass.
        Choosing 'fan_out' preserves the magnitudes in the backwards pass.
    nonlinearity (Literal['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu']) –
                the non-linear function (nn.functional name),
                recommended to use only with 'relu' or 'leaky_relu' (default).
                Why Specify Non-linearity?
                Specifying the nonlinearity allows the Kaiming formula to
                compensate for how the activation function alters the signal's
                variance. ReLU halves the variance (because it cuts negative
                inputs to zero), so Kaiming initialization compensates by
                increasing the initial weight variance by a factor of 2
                compared to Xavier.
    """
    def kaiming_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity='relu')
            #Note maxPool2d in nn.Conv2d has not bias.
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m.weight)
            nn.init.zeros_(m.bias)
        return

    def xavier_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        return

    def zeros_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        return

    def random_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight)
            nn.init.zeros_(m.bias)
        return

    initializer_dict = {
                        "kaiming": kaiming_init,
                        "xavier": xavier_init,
                        "zeros": zeros_init,
                        "random": random_init,
                        }
    return initializer_dict.get(init_type)

for name, model in zip(["Vanilla", "Kaiming", "Xavier", "Zeros", "Random"],
                        [VanillaCNNModel(),
                        VanillaCNNModel().apply(config_init("kaiming")),
                        VanillaCNNModel().apply(config_init("xavier")),
                        VanillaCNNModel().apply(config_init("zeros")),
                        VanillaCNNModel().apply(config_init("random"))]):
    print(f"___{name}___")
    train_test_loop(model, train_loader, test_loader)
