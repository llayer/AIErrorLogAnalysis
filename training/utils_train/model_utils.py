'''
    Contains functions used for model training, like:
    * EarlyStopping
    * EarlyStoppingFromMultipleMetrics
    * imblearn_sample
    * get_class_weights
    * DataPlaceholder
    * PredictDataCallback
'''

import numpy as np
import keras
from collections import deque


def imblearn_sample(X, Y, imblearn_class, verbose=0):
    '''
    @param X w/ shape (num_examples,num_error,num_sites) - where each number is integer of number of errors happened at that site
    @param Y w/ shape (num_examples,) - where each number is integer that represents one class
    @param imblearn_class - imblearn class for sampling ex: SMOTE(random_state=42, ratio=1.0)
    return X, Y - sampled from imblearn_class
    '''
    if verbose != 0: print('Before Upsampling: x:{}, y:{}'.format(X.shape, Y.shape))
    X_shape = X.shape
    X = X.reshape( X.shape[0], np.prod(X.shape[1:]) )
    X, Y = imblearn_class.fit_sample(X, Y)
    X = X.reshape( (len(X),) + X_shape[1:] )
    if verbose != 0: print('After Upsampling: x:{}, y:{}'.format(X.shape, Y.shape))
    return X, Y


def get_class_weights(Y, num_classes, inf=99): # Y is NOT one-hot encoded shape=(num_examples,)
    '''
    @param Y w/ shape (num_examples,) - where each number is integer that represents one class
    @param num_classes - number of classes in Y
    return array w/ shape (num_classes,) where each number is weight for that class (index = class number)
    '''
    class_counts = np.zeros(num_classes)
    indexes,counts = np.unique(Y,return_counts=True)
    class_counts[ indexes.astype(int) ] = counts # to prevent situations where there are no class examples
    total_count = np.sum(class_counts)
    results = (total_count/num_classes) * (1/class_counts)
    results[ np.isinf(results) ] = inf # replace infinity with some number (inf means there are no elements`)
    return results


class MultipleMetricsEarlyStopping(keras.callbacks.Callback):
    '''
        Does early stopping if all of the metric scores don't get better then best scores for n steps.
        First metric is most important, it cannot be worse then the best metric by more then p.
        where n is patience and p is percentage increase from best score.
    '''
    def __init__(self, metric_functions, modes, patience=7, moving_avg_length=2, max_percentage_delta=0.3):
        '''
        @param metric_functions - list w/ metric functions (y_true,y_pred) -> result
        @param modes - list w/ 'max' or 'min' string, represents to maximize/minimize
                       function from metric_function (lengths of both lists must be equal)
        @param patience - when training doesn't improve for n epochs, then stop. 
                          where n is this variable number
        @param max_percentage_delta - percentage that first metric can't diverge from 
                                      best score of that metric
        '''
        super().__init__()
        if len(metric_functions) != len(modes):
            raise(Exception('length doesnt match'))
        self.metric_functions = metric_functions
        self.best_scores = np.array([ -np.inf if m == 'max' else np.inf for m in modes ])
        self.is_better = np.array([ np.greater if m == 'max' else np.less for m in modes ])
        self.moving_avg_length = moving_avg_length
        self.moving_avg_buffer = deque(maxlen=moving_avg_length)
        self.importance = max_percentage_delta
        self.patience = patience
        self.wait = 0


    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0])) # (num_examples,num_classes)
        y_true = self.validation_data[1] # (num_examples,num_classes)
        current_scores = np.array([m(y_true, y_pred) for m in self.metric_functions])
        self.moving_avg_buffer.append( current_scores )
        if len(self.moving_avg_buffer) != self.moving_avg_length:
            scores = current_scores
        else:
            scores = np.mean(self.moving_avg_buffer, axis=0)
        bool_arr = np.array([ is_btr(s,b) for s,b,is_btr in zip(scores,self.best_scores,self.is_better) ])
        logs['main_score'] = current_scores[0]
        percentage_increase = (scores[0] - self.best_scores[0]) / self.best_scores[0]
        
        # see if atleast one is getting better
        if len(bool_arr[ bool_arr == True ])/len(bool_arr) != 0:
            self.wait = 0
            self.best_scores[ bool_arr ] = current_scores[ bool_arr ]
        else:
            self.wait += 1
        
        # see if waiting is bigger then patience
        if self.wait > self.patience:
            self.model.stop_training = True

        # see if main metric is not getting worse more then some percentage number self.importance
        first_metric_is_getting_worse = not self.is_better[0]( percentage_increase, 0 )
        percentage_is_bigger_then_importance = np.greater( np.abs(percentage_increase), self.importance )
        if first_metric_is_getting_worse and percentage_is_bigger_then_importance:
            self.model.stop_training = True


class DataPlaceholder():
    def __init__(self):
        class DataItem():
            def __init__(self):
                self.x = None
                self.y = None
        self.train = DataItem()
        self.val = DataItem()
        self.test = DataItem()


class PredictDataCallback(keras.callbacks.Callback):
    ''' saves predictions and labels into logs
    '''
    def __init__(self,model,x,y,log_word):
        self.x = x; self.y = y; 
        self.model = model
        self.log_word = log_word

    def on_epoch_end(self,epoch,logs={}):
        logs[self.log_word+'predictions'] = self.model.predict(self.x)
        logs[self.log_word+'labels'] = self.y


class EarlyStopping(keras.callbacks.Callback):
    '''
        Does early stopping on a single metric,
        if metric is not improving for n steps, then
        stop training. where n - patience
    '''
    def __init__(self, metric_fun, patience, mode='min'):
        '''
        @param metric_fun - function (y_true,y_pred) -> result
        @param patience - if metric is not improving for n steps, then
                          stop training. where n - this number
        @param mode - 'min' - minimize function or 'max' - maximize
        '''
        super().__init__()
        self.metric_fun = metric_fun
        self.patience = patience
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.wait = 0
        self.is_better = np.greater if mode == 'max' else np.less


    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0])) # (num_examples,num_classes)
        y_true = self.validation_data[1] # (num_examples,num_classes)
        score = self.metric_fun(y_true, y_pred)
        logs['early_stop_metric'] = score
        if self.is_better(score, self.best_score):
            self.wait = 0
            self.best_score = score
        else:
            self.wait += 1
        if self.wait > self.patience:
            self.model.stop_training = True
