"""
Reference: https://github.com/Exorust/TorchLeet/blob/main/torch/easy/quantize-lm/quantize-language-model.ipynb
Implement a language model using an LSTM and apply dynamic quantization to
optimize it for inference. Dynamic quantization reduces the model size and
enhances inference speed by quantizing the weights of the model.
Requirements
Define the Language Model:

Purpose: Build a simple language model that predicts the next token in a sequence.
Components:
Embedding Layer: Converts input tokens into dense vector representations.
LSTM Layer: Processes the embedded sequence to capture temporal (time/sequential) dependencies.
Fully Connected Layer: Outputs predictions for the next token.
Softmax Layer: Applies a probability distribution over the vocabulary for predictions.
Forward Pass:
Pass the input sequence through the embedding layer.
Feed the embedded sequence into the LSTM.
Use the final hidden state from the LSTM to make predictions via the fully connected layer.
Apply the softmax function to obtain probabilities over the vocabulary.
Apply Dynamic Quantization:

Quantize the model dynamically
Evaluate the quantized model's performance compared to the original model.


More reference about LTSM
https://d2l.ai/chapter_recurrent-modern/lstm.html
https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
word_embedding reference:
https://docs.pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp/

potential homework: Augmenting the LSTM part-of-speech tagger with character-level features on
https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic
from torch.serialization import add_safe_globals


# For ARM/mobile/Apple Silicon to use quantize_dynamic
torch.backends.quantized.engine = 'qnnpack'


class LanguageModel(nn.Module):
    """
    input parameter explanation in LSTM
    num_layers: how many LSTM cells in a row.
                The output of previous LSTM cell is the input of the next LSTM cell.
    “If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    batch_first = true:
               Input shape is (batch_size, seq_len, input_size). Most people naturally think of matches how we usually organize and understand data.
    batch_first = false:
               Input shape is (seq_len, batch_size, input_size). It is compatible with other layers to make integration smoother.

    output parameter of LTSM in def __init__
    sequential_output, hidden, cell = lstm output
    sequential_output: is in a shape of (batch_size, sequence_length, hidden_size).
                        the output carries for each batch, for each word in a sequence,
                        the sequential_output vector at [i th row in a batch, j th word in a sequence_length]
                        is a vector of the last layer of hidden state of LSTM of the j th word in a sequence of
                        the i th row in the batch.
    hidden : is in a shape of (num_layers, batch_size, hidden_size)
            the output carries for each layer of the LSTM, for each batch,
            the hidden vector at (i th layer of LSTM, j th row in a batch) is a hidden state at the ith layer
            for each element in the LAST WORD in a sequence in the j th row in a batch
    cell: is in a shape of (num_layers, batch_size, hidden_size)
            the output carries for each layer of the LSTM, for each batch,
            the hidden vector at (i th layer of LSTM, j th row in a batch) is a LSTM cell output
            at the ith layer in the LAST WORD in a sequence in the j th row in a batch
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_output, (hidden, cell) = self.lstm(x)
        x = self.softmax(self.fc1(lstm_output[:, -1, :]))
        return x


# Create synthetic training data
torch.manual_seed(42)
vocab_size = 50
seq_length = 10
batch_size = 32

X_train = torch.randint(0, vocab_size, (batch_size, seq_length)) #Random integer sequence input (mimics a sqeuenc of words)
y_train = torch.randint(0, vocab_size, (batch_size,)) #Random target words

# Initialize the model, loss function and optimizers
embed_size = 64
hidden_size = 128
num_layers = 2
model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs =  5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)

    loss.backward()
    optimizer.step()

    # Log progress every epoch
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

# Now, we will quantize the model dynamically to reduce its size and improve inference speed
# Quantization: Apply dynamic quantization to the language model
"""
Dynamic quantization is a model optimization technique that reduces model size and
speeds up inference by converting weights to lower-precision integers (e.g., int8)
while converting activations on-the-fly during runtime.
"""
quantized_model = quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype = torch.qint8 )

# save the quantized model
torch.save(quantized_model.state_dict(), "quantized_language_model.pth")

# Load the quantized model and test it
quantized_model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers)

#Apply dynamic quantization on the model after defining it.
quantized_model = quantize_dynamic(quantized_model, {nn.Linear, nn.LSTM}, dtype = torch.qint8)
quantized_model.load_state_dict(torch.load("quantized_language_model.pth", weights_only=False)) #We should set weights_only=True for better security. However, the True value does not work in my local machine.

#Testing the quantized model on a sample input
quantized_model.eval()
test_input = torch.randint(0, vocab_size, (1, seq_length))
with torch.no_grad():
    prediction = quantized_model(test_input)
    print(f"Prediction for input {test_input.tolist()}: {prediction.argmax(dim=1).item()}")
