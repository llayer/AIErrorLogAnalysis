import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.metrics import roc_curve, auc
import psutil

MEMORY_LIMIT = 80.0

# Manual check whether the virtual memory exceeds a threshold
def memory_kill(verbose = 0):
    
    mem = psutil.virtual_memory()
    if verbose == 1:
        print( 'Virtual memory:', mem[2])
    if mem[2] > MEMORY_LIMIT:
        return True
    else:
        return False


class FitControl(keras.callbacks.Callback):

    def __init__(self, val_gen = None, patience = 3, mode='min', multiinput = True, early_stopping = True, verbose = 0):

        super().__init__()
        self.validation_gen = val_gen 
        self.patience = patience
        self.verbose = verbose
        self.multiinput = multiinput
        self.early_stopping = early_stopping
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.wait = 0
        self.is_better = np.greater if mode == 'max' else np.less
        self.scores = []


    def on_epoch_end(self, epoch, logs={}):

        # Check memory
        if memory_kill(self.verbose) == True:
            self.model.stop_training = True
        
        # Calculate validation ROC
        if self.validation_gen is not None:
            
            y_targ_batches = []
            y_pred_batches = []
            for X,y in self.validation_gen.gen_batches():
                y_pred_batches.append(np.asarray(self.model.predict(X)))
                y_targ_batches.append(y)

            y_targ = np.concatenate(y_targ_batches)
            y_pred = np.concatenate(y_pred_batches)   
            
        else: 
            
            if self.multiinput == True:
                y_pred = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]])))
                y_targ = self.validation_data[2]
            else:
                y_pred = (np.asarray(self.model.predict(self.validation_data[0])))
                y_targ = self.validation_data[1]
            
            #y_pred_max = np.argmax(y_pred, axis=-1)
            
            
        #precision, recall, thresholds = precision_recall_curve(val_targ, val_predict)
        #monit = (auc(recall, precision), average_precision_score(val_targ, val_predict))
        
        score = roc_auc_score(y_targ, y_pred)
        self.scores.append(score)
        if self.verbose == 1:
            print('rocauc:', score )
        
        # Early stopping
        if self.early_stopping == True:
            if self.is_better(score, self.best_score):
                self.wait = 0
                self.best_score = score
            else:
                self.wait += 1
            if self.wait > self.patience:
                self.model.stop_training = True

           
            
            
            
            

