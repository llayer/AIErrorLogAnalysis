## Preprocessing

Preprocessing constist out of the steps: tokenization, cleaning, word2vec, indexing and creation of the input dataframe.

### 1. Tokenization

The tokenization step uses the NLTK wordbank tokenizer, considers only unique messages and saves them in chunks.

### 2. Cleaning

The cleaning step filters out low frequency words, special characters, cleans the tokens and saves the result in chunks.

