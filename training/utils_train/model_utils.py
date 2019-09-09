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
    if verbose > 0:
        print( 'Virtual memory:', mem[2])
    if mem[2] > MEMORY_LIMIT:
        return True
    else:
        return False


class FitControl(keras.callbacks.Callback):

    def __init__(self, train_gen = None, val_gen = None, patience = 3, mode='max', multiinput = True, early_stopping = True, 
                 store_best = False, store_best_roc = False, kill_slowstarts = False, kill_threshold = 0.51, verbose = 0):

        super().__init__()
        self.validation_gen = val_gen 
        self.training_gen = train_gen
        self.patience = patience
        self.verbose = verbose
        self.store_best = store_best
        self.store_best_roc = store_best_roc
        self.fpr_best = None
        self.tpr_best = None
        self.multiinput = multiinput
        self.early_stopping = early_stopping
        self.kill_slowstarts = kill_slowstarts
        self.kill_threshold = kill_threshold
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
            
        if self.training_gen is not None:
            
            y_targ_train_batches = []
            y_pred_train_batches = []
            for X,y in self.training_gen.gen_batches():
                y_pred_train_batches.append(np.asarray(self.model.predict(X)))
                y_targ_train_batches.append(y)

            y_targ_train = np.concatenate(y_targ_train_batches)
            y_pred_train = np.concatenate(y_pred_train_batches)  
            
            score_train = roc_auc_score(y_targ_train, y_pred_train)
            if self.verbose > 0:
                print('ROC train:', score_train )            
            
        #precision, recall, thresholds = precision_recall_curve(val_targ, val_predict)
        #monit = (auc(recall, precision), average_precision_score(val_targ, val_predict))
        
        score = roc_auc_score(y_targ, y_pred)
        self.scores.append(score)
        if self.verbose > 0:
            print('ROC test:', score )
        
        # Early stopping and saving
        if self.is_better(score, self.best_score):
            self.wait = 0
            self.best_score = score
            # Save the model
            if self.store_best == True:
                self.model.save(filepath = 'model.h5', overwrite=True)
            if self.store_best_roc == True:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                self.fpr_best = fpr
                self.tpr_best = tpr
        else:
            self.wait += 1
        if self.wait > self.patience and self.early_stopping == True:
            self.model.stop_training = True
        if self.kill_slowstarts == True and score < self.kill_threshold:
            self.model.stop_training = True
                

           
            
            
            
            

