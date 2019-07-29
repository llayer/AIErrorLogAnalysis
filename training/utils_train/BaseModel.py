'''
    Base model class
     - Main variables:
        self.model - keras model
        self.num_classes - number of outputs/classes
        self.class_weights - class weights got from get_class_weights()
                             used in weighted_cross_entropy
                             (if using imblearn this becomes simple cross_entropy)
                             
     - Main functions:
        self.create_model - creates self.model
        self.train - trains self.model w/ selected parameters
        self.predict - predicts Y from X with self.model
        self.change_inputs - function that changes inputs before feeding to model
        self.load_model - loads model.h5 file from directory (dirpath)
        self.save_model - saves model.h5 and model.json to directory (dirpath)
'''

import imblearn
import sklearn
import os
import json
import numpy as np
import keras
from tqdm import tqdm
import copy

from model_utils import DataPlaceholder, get_class_weights, \
                              imblearn_sample, PredictDataCallback
from model_utils import MultipleMetricsEarlyStopping
from metrics import accuracy, weighted_accuracy, recall, precision
from metrics import normalized_confusion_matrix_and_identity_mse as confusion_mse
from skopt_utils import get_best_params, create_skopt_results_string

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import gp_minimize
from skopt.utils import use_named_args


class Model():
    
    def __init__(self):
        self.model = None
        self.num_classes = None
        self.class_weights = None
        self.model_params = {}
        self.dimensions = None
        self.priors = None


    def create_model(self):
        ''' in self.train() this function is always called like:
            self.create_model(**self.model_params)
        '''
        pass


    def predict(self, X, argmax=True):
        ''' predicts Y from X with self.model
        @param X w/ shape (num_examples, num_error, num_sites)
        return y_argmax w/ shape (num_examples,)
        '''
        y_pred = self.model.predict(X) # (num_examples, num_outputs)
        if argmax:
            y_pred = np.argmax(y_pred, axis=-1) # (num_examples,)
        return y_pred

    
    def train(self, X, Y, max_epochs=100, batch_size=256, seed=42, verbose=0, # training vars
               use_imblearn=False, imblearn_class=SMOTE(random_state=42, ratio=1.0), # imblearn vars
               early_stopping_callback='default', test_split=0.2, # early stopping vars
               testing=False, kfold_function=KFold, kfold_splits=5, ): # testing vars (returns predictions from kfold)

        data = DataPlaceholder()

        if testing:
            enum = enumerate(kfold_function(n_splits=kfold_splits, shuffle=True, random_state=seed).split(X,Y))
            if verbose != 0:
                enum = tqdm(enum, total=kfold_splits, desc='kfold', leave=False, initial=0)
            histories = []
            for i,(index_train, index_valid) in enum:
                data.train.x, data.val.x = X[ index_train ], X[ index_valid ]
                data.train.y, data.val.y = Y[ index_train ], Y[ index_valid ]
                if use_imblearn:
                    data.train.x, data.train.y = imblearn_sample( data.train.x, data.train.y, imblearn_class, verbose=verbose )
                
                self.class_weights = get_class_weights(data.train.y, self.num_classes)

                self.create_model(**self.model_params)
                keras_callbacks = [ PredictDataCallback(self.model, data.train.x, data.train.y, ''),
                                    PredictDataCallback(self.model, data.val.x  , data.val.y  , 'val_') ]
                history = self.model.fit( x = data.train.x, y = data.train.y,
                                          validation_data = (data.val.x, data.val.y),
                                          nb_epoch = max_epochs, batch_size = batch_size,
                                          callbacks = [] + keras_callbacks,
                                          verbose = verbose )
                histories.append( history.history )
            return histories
        else:
            if early_stopping_callback == 'default':
                self.class_weights = get_class_weights(Y, self.num_classes)
                early_stopping_functions = [confusion_mse(), recall(average='macro'), precision(average='macro')]
                modes = ['min', 'max', 'max']
                early_stopping_callback = MultipleMetricsEarlyStopping(early_stopping_functions, modes=modes)
            
            data.train.x, data.val.x, data.train.y, data.val.y = train_test_split(X, Y, test_size=test_split, random_state=seed)
            if use_imblearn:
                data.train.x, data.train.y = imblearn_sample( data.train.x, data.train.y, imblearn_class, verbose=verbose )

            self.class_weights = get_class_weights(data.train.y, self.num_classes)
            data.train.x, data.train.y = self.change_inputs(data.train.x, data.train.y)
            data.val.x, data.val.y = self.change_inputs(data.val.x, data.val.y)
            self.create_model(**self.model_params)
            history = self.model.fit( x = data.train.x, y = data.train.y,
                                      validation_data = (data.val.x, data.val.y),
                                      epochs = max_epochs, batch_size = batch_size,
                                      callbacks = [early_stopping_callback], 
                                      verbose = verbose )
            return history.history

    
    def set_skopt_dimensions(self):
        ''' initializes self.dimensions list
            !!! order of elements must be the same as self.create_model() params !!!
            !!! name fields must be the same as keys in self.model_params dict   !!!
        '''
        pass


    def find_optimal_parameters( self, X, Y, num_calls=12, evaluation_function=confusion_mse(), # optimization vars
                                 test_split=0.3, max_epochs=100, batch_size=256, # training vars
                                 use_imblearn=False, imblearn_class=SMOTE(random_state=42, ratio=1.0), # imblearn vars
                                 early_stopping_callback='default', # early stopping vars (test split is the same)
                                 seed=42, verbose=0, summary_txt_path=None ): # other vars
        ''' finds optimal parameters, saves in self.model_params and returns them
        @param evaluation_function - function (y_true,y_pred)->float, where y_true,y_pred shapes are (num_examples,)
        @param summary_txt_path - path to where to save summary file
        @param num_calls - number of skopt optimization calls
        @param test_split - test split to evaluate model (the same data is used for early stopping, because seed is the same)
        !!! for other parameters see self.train() !!!
        return dict of best parameters {'param1':int, 'param2':int, ...}
        '''
        self.num_skopt_call = 0
        self.set_skopt_dimensions()
        prior_values = [a for a in self.model_params.values()]
        prior_names = [a for a in self.model_params.keys()]

        @use_named_args(self.dimensions)
        def fitness(**p): # p = { 'p1':0.1,'p2':3,... }
            global y_test
            if verbose != 0: print('\n \t ::: {} SKOPT CALL ::: \n'.format(self.num_skopt_call+1))
            print(p)
            self.model_params = p

            _, X_test, _, y_test = train_test_split(X, Y, test_size=test_split, random_state=seed)

            self.train( X, Y, max_epochs=max_epochs, batch_size=batch_size, use_imblearn=use_imblearn, 
                        imblearn_class=imblearn_class, early_stopping_callback=copy.deepcopy(early_stopping_callback), 
                        test_split=test_split, seed=seed, verbose=verbose )

            y_pred = self.predict(X_test,argmax=False)
            y_test = keras.utils.to_categorical(y_test)
            result = evaluation_function(y_test, y_pred)
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
        ''' loads model.h5 file from directory (dirpath)
        @param dirpath - path of directory, from where to load
        '''
        weightspath = os.path.join(dirpath,'model.h5')
        self.model.load_weights(weightspath)


    def save_model(self, dirpath):
        ''' saves model.h5 and model.json to directory (dirpath)
        @param dirpath - path of directory, where to save
        '''
        jsonpath = os.path.join(dirpath,'model.json')
        weightspath = os.path.join(dirpath,'model.h5')
        model_dict = json.loads(self.model.to_json())
        with open(jsonpath, "w") as json_file:
            json_file.write( json.dumps(model_dict,indent=4) )
        self.model.save_weights(weightspath)












