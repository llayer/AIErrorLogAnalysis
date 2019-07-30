import sklearn
import os
import json
import numpy as np
import keras
from tqdm import tqdm
import copy

from utils_train.model_utils import DataPlaceholder, get_class_weights, PredictDataCallback
from utils_train.model_utils import MultipleMetricsEarlyStopping
from utils_train.metrics import accuracy, weighted_accuracy, recall, precision
from utils_train.metrics import normalized_confusion_matrix_and_identity_mse as confusion_mse
from utils_train.skopt_utils import get_best_params, create_skopt_results_string

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import gp_minimize
from skopt.utils import use_named_args
from keras.callbacks import EarlyStopping


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
            
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
            keras_callbacks.append(es)
            
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
        print( keras_callbacks )
        
        history = self.model.fit( X_train, y_train, validation_data = (X_val, y_val), \
                                 epochs = max_epochs, batch_size = batch_size, callbacks = keras_callbacks)
        
        """
        history = self.model.fit( x = X_train, y = y_train,
                                  validation_data = (X_val, y_val),
                                  epochs = max_epochs, batch_size = batch_size,
                                  callbacks = [] + keras_callbacks,
                                  verbose = verbose )
        """
        
        return history.history
          
                                               

    def find_optimal_parameters( self, X_train, y_train, X_test, y_test, num_calls=12, evaluation_function=confusion_mse(), \
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

            early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
            
            self.train( X_train, y_train, X_test, y_test, max_epochs=max_epochs, batch_size=batch_size,  
                        early_stopping_callback=copy.deepcopy(early_stopping_callback), early_stopping = True, 
                        seed=seed, verbose=verbose )

            y_pred = self.predict(X_test,argmax=False)
            y_test_cat = keras.utils.to_categorical(y_test)
            result = evaluation_function(y_test_cat, y_pred)
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
    
    
    
    
    
    
    
    
    
    