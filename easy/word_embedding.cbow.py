"""
Reference: https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp/
https://www.geeksforgeeks.org/nlp/how-to-generate-word-embedding-using-bert/

There are two neural embedding methods for Word2Vec:
Continuous Bag of Words (CBOW) and Skip-gram.
"""
import torch
import torch.nn as nn
import torch.optim as optim

# Define CBOW model
# the i-context_size, ,..., the i-1 th, the i+1 th, ..., the i-context_size th
# words in a sentence to predict the i th word in the sentence.
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, vocab_size)
    def forward(self, context):
        # The output self.embeddings(context) is vocab_size x embed_size matrix.
        # Then squeeze it to be a vector containing embed size elements by summing all row value in each column.
        # Finally, transform it to be a tensor (1, embed_size) by using view (1,-1) or unsqueeze(0)
        context_embeds = self.embeddings(context).sum(dim=0).view(1,-1)
        y = self.relu(self.fc1(context_embeds))
        y = self.fc2(y)
        return y

context_size = 5
raw_text = "We downed the German outfit 3-1 over two legs to set up a clash with the Portuguese champions, who came from 3-0 down \
to win 5-3 on aggregate against Bodo/Glimt. The first leg of the quarter-finals will take place on Tuesday, April 7, at 8pm (UK time), \
at the Estadio Jose Alvalade. Our last visit to the stadium saw us prevail 5-1 in last season's league phase. The second leg will take \
place at Emirates Stadium on Wednesday, April 15, at 8pm (UK time). Both fixtures will be broadcast on TNT Sports. Ticket details for both \
fixtures will follow in due course. Stanford will face West Virginia in the quarterfinals on Thursday, April 2 at 5 p.m. in the eight-team bracket, \
and with a win would take on the winner of Creighton and Rutgers in the semifinals at 1 p.m. The four teams on the opposite \
side of the bracket are Oklahoma, Colorado, Minnesota and Baylor. We are thrilled to receive the at large Invitation to the \
College Basketball Crown, as we have heard nothing but great reviews about the tournament from last year,” said Kyle Smith, \
the Anne and Tony Joseph Director of Men’s Basketball. “We are getting an opportunity to compete against high major programs \
from the Big Ten, Big-12, SEC, and Big East, and FOX Sports gives us a platform to showcase our program and phenomenal freshman, \
Ebuka Okorie, on the national stage. The Cardinal is 0-1 against West Virginia all-time, with the only meeting coming in 1959 in \
Los Angeles. Stanford is 1-1 against both Creighton and Rutgers all-time, while it holds a 2-0 record this season against teams in \
the event with victories over Minnesota and Colorado during the non-conference slate."

tokens = raw_text.lower().split()
vocab = set(tokens)
word_to_index = { word:i for i, word in enumerate(vocab)}
data = []
for i in range(context_size, len(tokens) - context_size):
    # context is a sequence of words that having length 2 * context_size and the middle word is missing.
    context = [ word_to_index[word] for word in
                tokens[i-context_size: i] + tokens[i+1: i+context_size+1]
              ]
    target = word_to_index[tokens[i]]
    data.append((torch.tensor(context), torch.tensor(target)))

vocab_size = len(vocab)
embed_size = 64
learning_rate = 0.01
epochs = 100

cbow_model = CBOWModel(vocab_size, embed_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cbow_model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    total_loss = 0
    for context, target in data:
        output = cbow_model(context)
        # to make target from a tensor [] to tensor 1 by using unsqueeze(0).
        loss = criterion(output, target.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Example usage
"""
Note cbow_model.embeddings is a vocab_size X embedding_size matrix.
Thus, the ith row of the matrix is a vector having embedding size number of elements.
Hence, it is the word embedding of the matrix in the ith vocab.
"""
word_to_lookup = "The".lower()
word_index = word_to_index[word_to_lookup]
embedding = cbow_model.embeddings(torch.tensor(word_index)) #Note it mean the word_index th row of the embedding matrix
print(f"Embedding for {word_to_lookup}: {embedding.detach().numpy()}.")
