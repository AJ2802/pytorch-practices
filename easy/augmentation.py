"""
reference: https://github.com/Exorust/TorchLeet/blob/main/torch/easy/augmentation/augmentation.ipynb
You are tasked with applying data augmentation techniques to image data using torchvision.transforms. The goal is to enhance the variability of the input data by applying the following transformations:

Augmentation: generate more aritifical dataset from original limited dataset.

Random Horizontal Flip: Flip the image horizontally with a probability of 0.5. Horizontal flip is aka a mirror image along y-axis
Random Crop: Randomly crop the image to a specific size. If the size input is 16, it will crop a size 16 * 16 out of an image, e.g.
                                                             transforms.RandomCrop(size=16, padding = 4)
                                                         If the size=(16,8), it will crop a size 16 * 8 out of an image.
                                                             transforms.RandomCrop(size=(16,8), padding = 4)
Normalization: Normalize the image using a specified mean and standard deviation.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np

"""
Load CIFAR-10 dataset with data augmentation
"""
transform = transforms.Compose(
                                [
                                  transforms.ToTensor(),
                                  transforms.RandomHorizontalFlip(p=1),
                                  transforms.RandomCrop(size=(32,32), padding = 4), #the original image is 32 * 32. With padding, the iamge becomes 40*40. Thus, crop size is less than 40*40
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]
                            )

train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = False)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)

# Display a batch of augmented images
def imshow(img):
    #Note img.shape is 3*274*274. Not quite sure why it is 274 instead of 256 or 256+8
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    """
    convert an image object to numpy object.
    In this case, the output is a numpy 3 dimensional tensor.
    """
    print()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    """
    (1,2,0) indicate the 1st axis become the new 0th axis,
    the 2nd axis become the new 1st axis and
    the 0th axis becomes the new 3rd axis.
    In other words, flip the image along y-axis again. So it cancel randomHorizontal flip and shows the original image.
    A reason to put the 0th dim to the last. I believe it is coz the 0th dim is color dimension whose in numpy image format should be located to the last dimension.
    """
    plt.show()

# Get some random training images,
# Note each iteration of train_loader has 64 images coz of batch size is 64.
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Show images
imshow(torchvision.utils.make_grid(images))
