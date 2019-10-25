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


