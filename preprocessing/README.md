## Preprocessing

The preprocessing can be run with [input.py](https://github.com/llayer/AIErrorLogAnalysis/blob/master/preprocessing/input.py).
Preprocessing constist out of the following steps:
1. Tokenization: uses the NLTK wordbank tokenizer, considers only unique messages and saves them in chunks.
2. Cleaning: The cleaning step filters out low frequency words, special characters, cleans the tokens and saves the result in chunks.
3. Message selection: since there are multiple messages per key, only one message out of the sequence is chosen
4. word2vec: Running word2vec with the error messages creates an embedding matrix that can be used in the Keras embedding layer, either as initial weights or as fixed weights. Additionally a harder filtering / maximum number of words can be specified to reduce further the vocabulary.
5. Final input: In the final step the counts and labels from the actionshist are merged with the error messages. Both the averaged vector per message from word2vec is stored, as well as the indexed message for supervised training with RNNs.The output is a sparse pandas frame.

