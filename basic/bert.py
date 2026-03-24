"""
Reference: https://www.geeksforgeeks.org/nlp/how-to-generate-word-embedding-using-bert/

token : split each word of sentence into sub-words and encode them into IDs.
       e.g., "unbelievable"  ---> "un", "##believ", "##able"

Bert is a transformers model which can read text in both directions to analyze
all words in a sentence. It can be also fined-tuned by users  so that it can
meet to your NLP task while CBOW word embedding model cannot.

"""
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity



torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(random_seed)

# Load Bert tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained('bert-base-uncased')


# Input text
text = "GeeksforGeeks is a computer science portal"

# Tokenize and encode text using batch_encode_plus
# The function returns a dictionary containing the
# token IDs and attention masks.
encoding = tokenizer._encode_plus( [text], # List of input texts
                                         padding = True, # Pad to the maximum sequence length
                                         truncation = True, # Truncate to the maximum sequence length if necessary
                                         return_tensors = 'pt', # Return PyTorch tensors
                                         add_special_tokens = True # Add special tokens CLS and SEP
                                    )
# input_ids are numerical representatios
# of these tokens in text variable from
# Bert's vocabulary. The 6 words in a text
# becomes 12 tokens coming from Bert Vocabulary.
input_ids = encoding['input_ids']
print(f"Input IDs {input_ids}.")

# An attention_mask is a binary tensor (1s and 0s) that tells
# the model which tokens are real content (1) and which are
# padding (0), ensuring the model ignores filler data.
attention_mask = encoding['attention_mask']
print(f"Attention mask {attention_mask}.")

# Generate embeddings using Bert model
with torch.no_grad():
    outputs = model(input_ids , attention_mask = attention_mask )
    word_embeddings = outputs.last_hidden_state
    # the last hidden layer contains the embeddings.
    # word_embeddings is in shape of 1 x 12 x 768,
    #                                1 mean there is only one batch.
    #                               12 mean there are only 12 tokens in the input.
    #                               768 means the dimension of word embedding representation of a token.

# Output the shape of word embeddings
print(f"Shape of Word Embeddings: {word_embeddings.shape}")

#Decode the token IDs back to text
decoded_text = tokenizer.decode(input_ids[0], skip_special_token = True)
print(f"Decoded Text: {decoded_text}")

#Convert the decoded text back to tokens before converting them to token IDs
tokenized_text = tokenizer.tokenize(decoded_text)
print(f"tokenized Text: {tokenized_text}")

#encode the text into tokens after converting them to token IDs
encoded_text = tokenizer.encode(text, return_tensors = 'pt')
print(f"Encoded Text: {encoded_text}")

# Note word_embeddings is in a shape of 1 x 12 x 768
# So, word_embeddings[0] is in a shape of 12 x 768
for token, embedding in zip(tokenized_text, word_embeddings[0]):
    print(f"Embedding : {embedding}")
    print("\n")

# Compute the average of word embeddings to get the sentence embeddings
# Note mean over row coz word_embeddings is in shape of batch_size x row (dim = 1) x column
# So average along row mean average poolin along the whole sentence.
sentence_embedding = word_embeddings.mean(dim = 1)

# Print the sentence embedding
print("Sentence Embedding")
print(f"Shape of sentence embedding : {sentence_embedding.shape}")
print(sentence_embedding)

# Compute similarity metrics
example_sentence = "Greeksfor Greeks is a technology website."

# Tokenize and encode the example sentence
example_encoding = tokenizer._encode_plus(
                                [example_sentence],
                                padding = True,
                                truncation = True,
                                return_tensors = 'pt',
                                add_special_tokens = True
                                )

example_input_ids = example_encoding['input_ids']
example_attention_mask = example_encoding['attention_mask']

# Generate embeddings for the example sentence
with torch.no_grad():
    example_outputs = model(example_input_ids, attention_mask = example_attention_mask)
    example_sentence_embedding = example_outputs.last_hidden_state.mean(dim = 1)

# Compute cosine similarity between the original sentence embedding and the example sentence embedding
similarity_score = cosine_similarity(sentence_embedding, example_sentence_embedding)

print(f"Cosine Similarity Score: {similarity_score[0][0]}")
