'''
    *  Contains functions that invoked with or without parameters returns
    a function with inputs (y_true, y_pred), 
    y_true, y_pred shapes are (num_examples, num_classes)
    *  and a function get_metrics_functions, which returns dict of all metrics,
    where key = string name, value = metric function
'''

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, \
                            recall_score, precision_score, confusion_matrix, \
                            mean_squared_error
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
import keras.backend as K


def accuracy(average=True):
    ''' returns function that returns acc for each class. shape = (num_classes,)
    @param average if False then returns accuracy for all classes else averages them
    return function(y_true,y_pred) -> result 
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred, axis=-1)
        y_pred = to_categorical(y_argmax, num_classes=num_classes)
        accs = np.zeros(num_classes)
        for i in range(num_classes):
            accs[i] = accuracy_score(y_true[:,i],y_pred[:,i])
        if average:
            return np.mean(accs)
        else:
            return accs
    return metric


def weighted_accuracy(weights):
    ''' returns function that returns weighted accuracy (single number)
    return function(y_true,y_pred) -> result 
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        W = weights[ y_true ]
        return np.mean( W * (y_true == y_pred) )
    return metric


def recall(average='micro'):
    ''' returns recall
    @param average - 'micro' (dominant class is more important) (returns one number) or 
                     'macro' (smaller classes are more important) (returns one number) or 
                     None (no average, returns array w/ shape (num_classes,))
    return function(y_true,y_pred) -> result 
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred, axis=-1)
        y_pred = to_categorical(y_argmax, num_classes=num_classes)
        return recall_score(y_true, y_pred, average=average)
    return metric


def precision(average='macro'):
    ''' returns precision
    @param average - 'micro' (dominant class is more important) (returns one number) or 
                     'macro' (smaller classes are more important) (returns one number) or 
                     None (no average, returns array w/ shape (num_classes,))
    return function(y_true,y_pred) -> result 
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred, axis=-1)
        y_pred = to_categorical(y_argmax, num_classes=num_classes)
        return precision_score(y_true, y_pred, average=average)
    return metric


def roc_auc(average='macro'):
    ''' returns roc auc
    @param average - 'micro' (dominant class is more important) (returns one number) or 
                     'macro' (smaller classes are more important) (returns one number) or 
                     None (no average, returns array w/ shape (num_classes,))
    return function(y_true,y_pred) -> result 
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred, axis=-1)
        y_pred = to_categorical(y_argmax, num_classes=num_classes)
        return roc_auc_score(y_true, y_pred, average=average)
    return metric


def f1(average='macro'):
    ''' returns f1 score
    @param average - 'micro' (dominant class is more important) (returns one number) or 
                     'macro' (smaller classes are more important) (returns one number) or 
                     None (no average, returns array w/ shape (num_classes,))
    return function(y_true,y_pred) -> result 
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred, axis=-1)
        y_pred = to_categorical(y_argmax, num_classes=num_classes)
        return f1_score(y_true, y_pred, average=average)
    return metric


def conf_matrix(normalize=False):
    ''' returns function that returns confusion matrix 
        array w/ shape (num_classes,num_classes)
    @param normalize if True then normalizes matrix
    return function(y_true,y_pred) -> result
    '''
    def metric(y_true,y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)) # returns shape=(num_classes,num_classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm
    return metric


def normalized_confusion_matrix_and_identity_mse():
    ''' returns function that returns mse of 
        normalized confusion matrix and the identity matrix (single number)
    return function(y_true,y_pred) -> result 
    '''
    def metric(y_true,y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        cm = conf_matrix()(y_true,y_pred)
        num_classes = y_true.shape[-1]
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)) # returns shape=(num_classes,num_classes)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize confusion matrix
        cm = cm.flatten()
        cm[ np.isnan(cm) ] = 0 # replace nans w/ 0
        identity = np.identity(num_classes).flatten()
        mse = mean_squared_error(cm, identity)
        return mse
    return metric


def cross_entropy(e=1e-07, axis=-1):
    ''' returns cross entropy of all classes (single number)
        (must be one-encoded inputs) '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        y_pred = np.clip(y_pred, e, 1. - e)
        loss = y_true * np.log(y_pred)
        return - np.mean( np.sum(loss, axis=axis) )
    return metric


def weigthed_cross_entropy(weights, e=1e-07, axis=-1):
    ''' returns weighted cross entropy of all classes (single number)
        (must be one-encoded inputs) '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        y_pred = np.clip(y_pred, e, 1. - e)
        loss = y_true * np.log(y_pred) * weights
        return - np.mean( np.sum(loss, axis=axis) )
    return metric
