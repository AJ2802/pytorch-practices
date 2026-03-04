import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic sequential data
torch.manual_seed(42)
sequence_length = 10
num_samples = 100

#Create a sine wave dataset
X = torch.linspace(0, 4 * 3.14159, steps = num_samples).unsqueeze(1)
y = torch.sin(X)

# Prepare data for RNN
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i+seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)
"""
e.g. data = [1, 2, ,3 ,4, 5, 6, 7, 8, 9, 10], seq_length = 3
torch.stack(in_seq) and torch.stack(out_seq)
    [1,2,3]                 [4]
    [2,3,4]                 [5]
    [4,5,6]                 [6]
    [5,6,7]                 [7]
    [6,7,8]                 [8]
    [7,8,9]                 [9]
    [8,9,10]                [10]
"""

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

"""
RNN model
if an sequence input is [1,2,3]
# 1 x 50 vector is made from a fully connected graph from the input with weight W_ih and the bias
fully connect graph from the previous 1 x 50 vector with hidden weight W_hh and the same bias

# 1 x 50 vector is made from a fully connected graph from the input with the same weight W_ih
fully connect graph from the previous 1 x 50 vector with the same hidden weight W_hh

# 1 x 50 vector is made from a fully connected graph from the input with the same weight W_ih

    fully linear                activation
    with the same               tanh
    W_ih + bias b_h
[1]     --->        [....]      --->        [....] # 1 x 50 vector vector values after activation
                       ________________________|
                       | the linear part is added by a 1 x 50 vector from a fully connected graph of activation value with the same hidden weight W_hh
                       \/
[2]     --->        [....]      --->        [....]
                       ________________________|
                       | the linear part is added by a 1 x 50 vector from a fully connected graph of activation value with the same hidden weight W_hh
                       \/
[3]     --->        [....]      --->        [....]

Use the last  1x50 vector after activation as an input of a fully connected layer aka output layer in this example.
"""
class RNNModel(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 50, output_dim = 1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Weight matrices for input and hidden state
        self.W_ih = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        #Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        #Activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h_t = torch.zeros(batch_size, self.hidden_dim, device = x.device)

        for t in range(seq_len):
            x_t = x[:,t,:]
            #Perform matrix multiplication using the @ operator
            #recurrent network:
            #use past values h_t (1 x 50 vector) as an input + original input x to acts on a current layer.
            h_t = self.tanh(x_t @ self.W_ih + h_t @ self.W_hh + self.b_h)

        output = self.output_layer(h_t)
        return output

model = RNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)

epochs = 300
for epoch in range(epochs):
    for sequences, labels in zip(X_seq, y_seq):
        sequences = sequences.unsqueeze(0) # Add batch dimension. Make sequences.shapes changes from 10 x 1 to 1 x 10 x 1

        labels = labels.unsqueeze(0) # Add batch dimension.  Make sequences.shapes changes from 10 x 1 to 1 x 10 x 1

        pred = model(sequences)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing on new data
X_test = torch.sin(torch.linspace(4 * 3.14159, 8*3.14159, steps = 100).unsqueeze(1))
# Reshape to (batch_size, sequence_length, input_size)
X_test = X_test.unsqueeze(0) # Add batch dimension, shape becomes (1, 100, 1)

with torch.no_grad():
    print(f"The last 4 values of X_test: {X_test[:, -4: ,:].tolist()}")
    print(f"Predict the last values of X_test based on its previous sequence")
    print(f"Input is X_test[,:-4,:] and the output is {model(X_test[:, :-4 ,:]).item():.4f}")
    print(f"Input is X_test[,:-3,:] and the output is {model(X_test[:, :-3 ,:]).item():.4f}")
    print(f"Input is X_test[,:-2,:] and the output is {model(X_test[:, :-2,:]).item():.4f}")
    print(f"Input is X_test[,:-1,:] and the output is {model(X_test[:, :-1,:]).item():.4f}")
    print("They all match to the above last 4 values of X_test")
