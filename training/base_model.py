import sklearn
import os
import json
import numpy as np
import keras
from tqdm import tqdm
import copy

from utils_train.model_utils import DataPlaceholder, get_class_weights, PredictDataCallback
#from utils_train.model_utils import MultipleMetricsEarlyStopping, EarlyStopping
from utils_train.metrics import accuracy, weighted_accuracy, recall, precision, roc_auc
from utils_train.metrics import normalized_confusion_matrix_and_identity_mse as confusion_mse
from utils_train.skopt_utils import get_best_params, create_skopt_results_string

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import gp_minimize
from skopt.utils import use_named_args
from keras.callbacks import EarlyStopping

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

class Metrics(Callback):
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0])))
        val_targ = self.validation_data[1]
        #_val_f1 = f1_score(val_targ, val_predict, average='weighted')
        #_val_recall = recall_score(val_targ, val_predict, average='weighted')
        #_val_precision = precision_score(val_targ, val_predict, average='weighted')
        _val_rocauc = roc_auc_score(val_targ, val_predict)
        
        #roc = roc_auc()
        #_val_rocauc = roc(val_targ, val_predict)
        #self.val_f1s.append(_val_f1)
        #self.val_recalls.append(_val_recall)
        #self.val_precisions.append(_val_precision)
        print( _val_rocauc )
        #print( " — val_f1: %f — val_precision: %f — val_recall %f - val_rocauc %f" \
        #      %(_val_f1, _val_precision, _val_recall, _val_rocauc) )
        return

metrics = Metrics()

"""
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
"""

class BaseModel():
    
    def __init__(self, use_batch_gen = False):
        self.model = None
        self.num_classes = None
        self.class_weights = None
        self.model_params = {}
        self.dimensions = None
        self.priors = None
        self.use_batch_gen = use_batch_gen


    def create_model(self):
        pass
    

    def predict(self, X, argmax=True):

        y_pred = self.model.predict(X) # (num_examples, num_outputs)
        if argmax:
            y_pred = np.argmax(y_pred, axis=-1) # (num_examples,)
        return y_pred
    
    
    
    def set_skopt_dimensions(self):
        pass

    
    def train_on_batch(self, training_generator, validation_generator, \
                       epochs = 100, steps_per_epoch = 100, validation_steps = 100,  \
                       early_stopping_callback='default', early_stopping=False):
        
        #self.class_weights = get_class_weights(data.train.y, self.num_classes)
        self.create_model(**self.model_params)
        
        self.print_summary()
        
        """
        keras_callbacks = [ PredictDataCallback(self.model, X_train, y_train, ''),
                            PredictDataCallback(self.model, X_val  , y_val  , 'val_') ]
        """
        
        if early_stopping == True:
            
            if early_stopping_callback == 'default':
                early_stopping_functions = [confusion_mse(), recall(average='macro'), precision(average='macro')]
                modes = ['min', 'max', 'max']
                early_stopping_callback = MultipleMetricsEarlyStopping(early_stopping_functions, modes=modes)
                kears_callbacks.append(early_stopping_callback)    
        
        history = self.model.fit_generator(generator=training_generator.gen_inf_count_msg_batches(), \
                                           steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, \
                                           validation_data=validation_generator.gen_inf_count_msg_batches(), 
                                           use_multiprocessing = True)

        return history.history           
        
    
    def train(self, X_train, y_train, X_val, y_val, max_epochs=100, batch_size=256, seed=42, verbose=0, \
              early_stopping_callback='default', early_stopping=False): 

        
            
        self.class_weights = get_class_weights(y_train, self.num_classes)
        self.create_model(**self.model_params)
            
        
        keras_callbacks = [ PredictDataCallback(self.model, X_train, y_train, ''),
                            PredictDataCallback(self.model, X_val  , y_val  , 'val_') ]
        
        if early_stopping == True:
            
            print( 'Set early stopping' )
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
            #es = EarlyStopping( roc_auc(), 5, 'max' )
            #keras_callbacks.append(es)
            
            """
            if early_stopping_callback == 'default':
                early_stopping_functions = [confusion_mse(), recall(average='macro'), precision(average='macro')]
                modes = ['min', 'max', 'max']
                early_stopping_callback = MultipleMetricsEarlyStopping(early_stopping_functions, modes=modes)
                keras_callbacks.append(early_stopping_callback)
            """
        
        self.model.summary() 
        print(X_train.shape)
        print(X_val.shape)
        
        
        #keras_callbacks.append( roc_callback(training_data=(X_train, y_train),validation_data=(X_val, y_val)))
        
        #print( keras_callbacks )
        
        history = self.model.fit( X_train, y_train, validation_data = (X_val, y_val), \
                                 epochs = max_epochs, batch_size = batch_size, callbacks = [es] )#[metrics] + keras_callbacks)
        
        """
        history = self.model.fit( x = X_train, y = y_train,
                                  validation_data = (X_val, y_val),
                                  epochs = max_epochs, batch_size = batch_size,
                                  callbacks = [] + keras_callbacks,
                                  verbose = verbose )
        """
        
        return history.history
          
                                               

    def find_optimal_parameters( self, X_train, y_train, X_test, y_test, num_calls=12, evaluation_function=roc_auc(), \
                                 max_epochs=100, batch_size=256, \
                                 early_stopping_callback='default', \
                                 seed=42, verbose=1, summary_txt_path=None ): 
        
        self.num_skopt_call = 0
        self.set_skopt_dimensions()
        prior_values = [a for a in self.model_params.values()]
        prior_names = [a for a in self.model_params.keys()]

        @use_named_args(self.dimensions)
        def fitness(**p): # p = { 'p1':0.1,'p2':3,... }
            #global y_test
            #y_test = y_t
            if verbose != 0: print('\n \t ::: {} SKOPT CALL ::: \n'.format(self.num_skopt_call+1))
            print(p)
            self.model_params = p

            early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            #EarlyStopping( roc_auc(), 5 )
            #early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
            
            self.train( X_train, y_train, X_test, y_test, max_epochs=max_epochs, batch_size=batch_size,  
                        early_stopping_callback=copy.deepcopy(early_stopping_callback), early_stopping = True, 
                        seed=seed, verbose=verbose )

            y_pred = self.predict(X_test,argmax=False)
            y_test_cat = keras.utils.to_categorical(y_test)
            #result = evaluation_function(y_test_cat, y_pred)
            result = -1 * roc_auc_score(y_test, y_pred)
            if verbose != 0: print('Result: {}'.format(result))
            self.num_skopt_call += 1
            return result
        
        search_result = gp_minimize( func = fitness, dimensions = self.dimensions,
                                     acq_func = 'EI', # Expected Improvement.
                                     n_calls = num_calls, x0 = prior_values )
        
        s = create_skopt_results_string( search_result, prior_names, 
                                         num_calls, summary_txt_path )
        if verbose != 0: print(s)
        
        best_params = get_best_params( search_result, prior_names )
        self.model_params = best_params
        return best_params
    
    
    def load_model(self, dirpath):

        weightspath = os.path.join(dirpath,'model.h5')
        self.model.load_weights(weightspath)


    def save_model(self, dirpath):

        jsonpath = os.path.join(dirpath,'model.json')
        weightspath = os.path.join(dirpath,'model.h5')
        model_dict = json.loads(self.model.to_json())
        with open(jsonpath, "w") as json_file:
            json_file.write( json.dumps(model_dict,indent=4) )
        self.model.save_weights(weightspath)
    
    
    
    
    
    
    
    
    
    