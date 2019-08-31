## Preprocessing

Preprocessing constist out of the steps: tokenization, cleaning, word2vec, indexing and creation of the input dataframe.

### 1. Tokenization

The tokenization step uses the NLTK wordbank tokenizer, considers only unique messages and saves them in chunks.

### 2. Cleaning

The cleaning step filters out low frequency words, special characters, cleans the tokens and saves the result in chunks.

### 3. Message selection

Since there are multiple messages per key, only one message out of the sequence should be chosen

### 4. word2vec

Running word2vec with the error messagescreates an embedding matrix that can be used in the Keras embedding layer, either as initial weights or as fixed weights. Additionally a harder filtering / maximum number of words can be specified to reduce further the vocabulary.

### 5. Final input
In the final step the counts and labels from the actionshist are merged with the error messages. The output is a sparse pandas frame that can subsequently be used for the training.

