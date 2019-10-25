# AIErrorLogAnalysis with Natural Language Processing
Repository for the analysis of log files of failing workflows of the CMS Experiment.
The goal is to predict the operator's actions for failing workflows stored in the [WTC Console](https://github.com/CMSCompOps/WorkflowWebTools). The input for the machine learning are the error logs
of the failing jobs and information about the frequency of the error per site.


## DAQ of the error messages

To run the analysis of the [WMArchive](https://github.com/dmwm/WMArchive) entries with Apache Spark on SWAN and filter the error log snippets run the notebook [filter_wm.ipynb](https://github.com/llayer/AIErrorLogAnalysis/blob/master/spark/filter_wm.ipynb).
The recommended options for SWAN are:
```
Software Stack: Bleeding Edge
Memory: 10Gb
Spark cluster: General Purpose (Analytix)
```
The output is saved in pandas frames.


## Preprocessing

The preprocessing can be run with [input.py](https://github.com/llayer/AIErrorLogAnalysis/blob/master/preprocessing/input.py).
Preprocessing constist out of the following steps:
1. Tokenization: uses the NLTK wordbank tokenizer, considers only unique messages and saves them in chunks.
2. Cleaning: The cleaning step filters out low frequency words, special characters, cleans the tokens and saves the result in chunks.
3. Message selection: since there are multiple messages per key, only one message out of the sequence is chosen
4. word2vec: Running word2vec with the error messages creates an embedding matrix that can be used in the Keras embedding layer, either as initial weights or as fixed weights. Additionally a harder filtering / maximum number of words can be specified to reduce further the vocabulary.
5. Final input: In the final step the counts and labels from the actionshist are merged with the error messages. Both the averaged vector per message from word2vec is stored, as well as the indexed message for supervised training with RNNs.The output is a sparse pandas frame.


## Training of the neural networks

### Baseline model
To reproduce the previous results without NLP the training can be run with 
[train_baseline.py](https://github.com/llayer/AIErrorLogAnalysis/blob/master/training/train_baseline.py).
For the hyperparameter optimization Bayesian optimization with scikit-optimize is used.

### NLP model 
To train a single NLP model run [train.py](https://github.com/llayer/AIErrorLogAnalysis/blob/master/training/train.py). 
For the hyperparameter optimization there are three options:
1. Run scikit-optimize with multiple threads [threaded_skopt.py](https://github.com/llayer/AIErrorLogAnalysis/blob/master/training/threaded_skopt.py)
2. Run on SWAN with spark (experimental) [train_on_spark.ipynb](https://github.com/llayer/AIErrorLogAnalysis/blob/master/training/train_on_spark.ipynb)
The following options should be chosen for SWAN:
```
Software Stack: Bleeding Edge
Memory: 10Gb
Spark cluster: Cloud Containers (Kubernets)
```
and spark:
```
spark.dynamicAllocation.enabled=False
spark.executor.instances = n (you can have up to 60)
spark.executor.memory 12g (you can specify even 14-15g if you want)
spark.executor.cores 3
spark.kubernetes.executor.request.cores 3
```
3. Train the model distributed on multiple GPUs with the NNLO framework (experimental) [example_nlp.py](https://github.com/llayer/NNLO/blob/master/examples/example_nlp.py)
