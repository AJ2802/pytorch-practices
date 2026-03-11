"""
https://github.com/Exorust/TorchLeet/blob/main/torch/easy/autoencoder/autoencoder.ipynb

You are tasked with implementing an autoencoder model
for anomaly detection. The model will be trained on the
MNIST dataset, and anomalies will be detected based on
the reconstruction error. The autoencoder consists of an
encoder to compress the input and a decoder to reconstruct
the image. The difference between the original image and the
reconstructed image will be used to detect anomalies.

The autoencoder should work on the MNIST dataset, which
consists of 28x28 grayscale images.

For upsampling in decoder, please check
https://www.geeksforgeeks.org/machine-learning/apply-a-2d-transposed-convolution-operation-in-pytorch/
https://d2l.ai/chapter_computer-vision/transposed-conv.html

Note:
ConvTranspose2d vs Conv2d
Conv2d does not increases the pixels.
ConvTranspose2d increases the pixels. (upsampling)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5), (0.5))     #Normalization transform actually make autoencoder hard to learn MNIST.
                ])
train_dataset = torchvision.datasets.MNIST( root="./data", train=True, download=True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size=64)

test_dataset = torchvision.datasets.MNIST( root="./data", train=False, download=True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size=64)

#Define an Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            #output shape is 16*28*28
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1 ),
            nn.ReLU(),

            #output shape is 16*14*14
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            #output shape is 32*14*14
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1 ),
            nn.ReLU(),

            #output shape is 32*7*7
            nn.MaxPool2d(kernel_size = 2, stride = 2)

        )
        """
        For ConvTranspose2d in decoder:
        output shape is 32*28*28 kernel_size + stride*(number of pixels + 2*padding - 1) + 2*output_padding

        padding in ConvTranspose2d is to remove the row and column starting from the most outerlayer of output.
        It is a reverse of a padding in Conv2d.
        But it make sense. Coz if we add padding size in the output of Conv2d,
        then we need to remove them in the output of ConvTranspose2d

        output_padding is the add zero value vector to right column and the bottom of the output layer.
        """
        self.decoder = nn.Sequential(
            # the output pixel in height/width is 3 + 2*(7-1) - 2 + 1 = 14 and the shape is 16*14*14
            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1 ),
            nn.ReLU(),

            # the output pixel in height/width is 3 + 2*(14-1) - 2 + 1 = 28 and the shape is 1*28*28
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1 ),
            nn.Sigmoid() # coz we should retun a pixel values which is between 0 and 1.
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-05)

epochs = 5
for epoch in range(epochs):
    for images, _ in train_loader:
        decoded_images = model(images)
        loss = criterion(decoded_images, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

"""
model.eval() is a kind of switch for some specific
layers/parts of the model that behave differently
during training and inference (evaluating) time.
For example, Dropouts Layers, BatchNorm Layers etc.
You need to turn them off during model evaluation,
and .eval() will do it for you. In addition,
the common practice for evaluating/validation
is using torch.no_grad() in pair with model.eval()
to turn off gradients computation

BUT, don't forget to turn back to training mode after
eval step: , e.g.
# training step
model.train()

Reference: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
"""
# Detect anomalies using reconstruction error
threshold = 0.1 
model.eval()
anomalies = []
normal = []
with torch.no_grad():
    for images, _ in test_loader:
        reconstructed_images = model(images)
        loss = criterion(reconstructed_images, images)
        print("loss.item() > threshold ",loss.item() > threshold)
        if loss.item() > threshold:
            anomalies.append(images)
        else:
            normal.append(images)

# # Visualize normal
# if normal:
#     # Select the first anomaly image in the first batch having anomaly images.
#
#     # remove the channel dimension
#     for image in normal[0]:
#         normal_image = image.squeeze()
#
#         print(f"Unanomaly image shape: {normal_image.shape}")
#         # Convert tensor to NumPy array for visualization
#         plt.imshow(normal_image.cpu().numpy(), cmap="gray")
#         plt.show()
# else:
#     print("No unanomalies detected.")

# Visualize anomalies
if anomalies:
    # Select the first anomaly image in the first batch having anomaly images.

    # remove the channel dimension
    for image in anomalies[0]:
        anomaly_image = image.squeeze()

        print(f"Anomaly image shape: {anomaly_image.shape}")
        # Convert tensor to NumPy array for visualization
        plt.imshow(anomaly_image.cpu().numpy(), cmap="gray")
        plt.show()
else:
    print("No anmalies detected.")
