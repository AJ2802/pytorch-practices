"""
Reference:https://github.com/Exorust/TorchLeet/blob/main/torch/basic/tensorboard/tensorboard.ipynb
Using TensorBoard to monitor the training progress of a linear
regression model in PyTorch. TensorBoard provides a visual
interface to track metrics such as loss during training,
making it easier to analyze and debug your model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Generate synthetic data
torch.manual_seed(42)
X = torch.rand(100, 1)*10 # 100 data points between 0 and 10.
y = 3 * X + 5 + torch.randn(100, 1) # Linear relationship with noise.

# Define a simple Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1) # Single input and single output

    def forward(self, x):
        return self.linear(x)

writer = SummaryWriter()

# Initialize the mdoel, loss function and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 1000
for epoch in range(epochs):
    y_pred = model(X) #same as model.forward(X)
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Log loss to TensorBoard
    writer.add_scalar('Loss value', loss.item(), epoch)

    #Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss:{loss.item():.4f}")

# Close the TensorBoard write
writer.close()

# Run TensorBoard using the logs generated
"""
open a terminal
install tensorboard
    pip install tensorboard
then enter
    tensorboard --logdir=runs
then copy the url in the output and view it on a browser.

NOTE: writer in pytorch always saved its log file in a default folder called runs.
the --logdir=runs mean recursively search a log file in a runs folder.
"""
