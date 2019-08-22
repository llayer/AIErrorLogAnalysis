import os
import shutil
import keras
import pandas as pd
import numpy as np
import psutil
import setGPU
from models import baseline_model
from models import w2v_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from models import nlp_model
from keras import backend as K
from skopt.utils import use_named_args
from skopt import gp_minimize
from utils_train import model_utils
from data_loader import input_batch_generator
from data_loader import index
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight

"""
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
"""

class FitHandler(object):
    

    def __init__(self, model_type, codes, sites, embedding_dim, gen_param, pruning_mode = 'None', nlp_param = None,
                 train_on_batch = True, use_smote = False, store = True, name = 'test', overwrite = True, verbose = 1):
        
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.train_on_batch = train_on_batch
        self.gen_param = gen_param
        self.use_smote = use_smote
        self.store = store
        self.path = '/nfshome/llayer/AIErrorLogAnalysis/results/' + name 
        self.verbose = verbose
        self.nlp_param = nlp_param
        self.sites_index, self.codes_index = self.prune( codes, sites, pruning_mode )
        self.dim_sites = len(list(set(self.sites_index.values())))
        self.dim_errors = len(list(set(self.codes_index.values())))
        mem = psutil.virtual_memory()
        
        if self.store == True:
            self.create_dir(overwrite)
        
        print( 'Memory:', mem[2] )
        print( 'Errors:', self.dim_errors, 'Sites:', self.dim_sites, 'Embedding dim:', self.embedding_dim )
        print( 'Pruning:', pruning_mode )
        print( 'Model:', self.model_type )
        print( self.gen_param )

        
    def get_model(self):
        
        if self.model_type == 'baseline':
            return baseline_model.FF(2, self.dim_errors, self.dim_sites)
        
        if self.model_type == 'nlp_w2v_counts':
            return w2v_model.ErrorSiteAverage(2, self.dim_errors, self.dim_sites, self.embedding_dim)
        
        if self.model_type == 'nlp_w2v':
            return w2v_model.W2V(2, self.dim_errors, self.dim_sites, self.embedding_dim)        

        if self.model_type == 'nlp_msg':
            if self.nlp_param is not None:
                return nlp_model.NLP(2, self.dim_errors, self.dim_sites, self.embedding_dim, 
                                     cudnn = self.nlp_param['cudnn'],
                                     batch_norm = self.nlp_param['batch_norm'], 
                                     train_embedding = self.nlp_param['train_embedding'], 
                                     word_encoder = self.nlp_param['word_encoder'], 
                                     encode_sites = self.nlp_param['encode_sites'],
                                     include_counts = self.nlp_param['include_counts'], 
                                     attention = self.nlp_param['attention'] ) 
            else:
                return nlp_model.NLP(2, self.dim_errors, self.dim_sites, self.embedding_dim ) 
        
        
    def create_dir(self, overwrite):
        
        if overwrite == True:
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
                os.makedirs(self.path)
            else:
                os.makedirs(self.path)
            print( 'Create directory:', self.path )
        else:
            try:
                os.makedirs(self.path)
                print( 'Create directory:', self.path )
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise        
    
    
    def print_summary_labels(self, X):
        
        print( X['label'].value_counts() )
    
    
    def print_model(self, model_param=None):
        
        model = self.get_model()
        if model_param is not None:
            model.set_hyperparameters(model_param)
        model.create_model()
            
        model.model.summary()
        
        del model
        K.clear_session()
        
        
    def check_batch(self, X, batch_size, start, stop):
        
        generator = input_batch_generator.InputBatchGenerator(X, 'label', self.codes_index, self.sites_index,
                                                              self.embedding_dim, batch_size = batch_size, 
                                                              averaged=self.gen_param['averaged'], 
                                                              first_only=self.gen_param['first_only'],
                                                              only_msg = self.gen_param['only_msg'], 
                                                              max_msg = self.gen_param['max_msg'])   
        return generator.msg_batch(start, stop)
    
    
    def prune(self, codes, sites, mode):
                
        if mode == 'None':
            sites_index, codes_index = index.to_index(list(sites['site']), list(codes['error']))
        elif mode == 'Tiers':
            _, codes_index = index.to_index(list(sites['site']), list(codes['error']))
            sites_index = index.tiers_to_index(list(sites['site']))
        elif mode == 'Neg':
            sites_index, codes_index = index.prune_to_index(codes, sites, only_unknown = True)
        elif mode == 'LowFreq':
            sites_index, codes_index = index.prune_to_index(codes, sites, only_unknown = True,
                                                    counts = False, error_threshold = 100, site_threshold = 1000)
        else:
            print( 'No valid indexing' )
        
        return sites_index, codes_index
                                                                                   
    
    def smote(self, X, y):
        
        smt = SMOTE()
        X_shape = X.shape
        X = X.reshape( X.shape[0], np.prod(X.shape[1:]) )
        X, y = smt.fit_sample(X, y)
        X = X.reshape( (len(X),) + X_shape[1:] )
        return X, y
    
    
    def test_training(self, X, y=None, model_param = None, batch_size = 10, max_epochs = 1, test_size=0.33):
        
        if y is None:
            X_train, X_test = self.split(X, test_size = test_size)
            self.train_in_batches(X_train, X_test, batch_size, model_param = model_param, 
                                  batch_size_val = batch_size, max_epochs = max_epochs)
        else:
            X_train, X_test, y_train, y_test = self.split(X, y, test_size = test_size)              
            self.train( X_train, y_train, X_test, y_test, max_epochs = max_epochs, batch_size = batch_size, 
                       model_param = model_param, early_stopping = True)            
    
    
    def split(self, X, y=None, test_size=0.33):
        
        if y is not None:
            return train_test_split(X, y, test_size = test_size,  stratify = y)   
        else:
            return train_test_split(X, test_size = test_size,  stratify = X['label'])
    
    
    def count_matrix(self, X):
        
        generator = input_batch_generator.InputBatchGenerator(X, 'label', self.codes_index, self.sites_index, self.embedding_dim)
        return generator.get_counts_matrix()
    
        
    def train_in_batches(self, X_train, X_test, batch_size_train, model_param = None, batch_size_val = 10, max_epochs = 20,
                         early_stopping = True):
            
        
        class_weights = class_weight.compute_class_weight('balanced', np.unique( X_train['label']), X_train['label'])
        
        model = self.get_model()
        if model_param is not None:
            model.set_hyperparameters(model_param)
        model.create_model()
          
        steps_per_epoch = int(len(X_train) / batch_size_train)
        steps_val = int(len(X_test) / batch_size_val)

        generator_train = input_batch_generator.InputBatchGenerator(X_train, 'label', self.codes_index, self.sites_index,
                                                                    self.embedding_dim, batch_size = batch_size_train, 
                                                                    averaged=self.gen_param['averaged'], 
                                                                    first_only=self.gen_param['first_only'],
                                                                    only_msg = self.gen_param['only_msg'], 
                                                                    max_msg = self.gen_param['max_msg'])

        generator_test = input_batch_generator.InputBatchGenerator(X_test, 'label', self.codes_index, self.sites_index,
                                                                   self.embedding_dim, batch_size = batch_size_val, 
                                                                   averaged = self.gen_param['averaged'], 
                                                                   first_only= self.gen_param['first_only'],
                                                                   only_msg = self.gen_param['only_msg'], 
                                                                   max_msg = self.gen_param['max_msg'] )
        
        if self.gen_param['only_msg'] == True:
            multiinput = False
        else:
            multiinput = True
        
        control_callback = model_utils.FitControl(train_gen = generator_train, val_gen = generator_test, 
                                                  mode = 'max', multiinput = multiinput, verbose=self.verbose)
        model.model.fit_generator(generator = generator_train.gen_inf_batches(), steps_per_epoch = steps_per_epoch,
                                  callbacks = [control_callback], epochs = max_epochs, class_weight = class_weights,
                                  verbose = self.verbose)
        
        score = control_callback.best_score
        print( 'Final score', score )
        
        del model
        K.clear_session() 
        
        return score        
        
        

    def train(self, X_train, y_train, X_test, y_test, model_param = None, batch_size = 100, 
              max_epochs = 200, early_stopping = True):
        
        if self.use_smote == True:
            X_train, y_train = self.smote(X_train, y_train)
        
        class_weights = class_weight.compute_class_weight('balanced', np.unique( y_train ), y_train)

        model = self.get_model()
        if model_param is not None:
            model.set_hyperparameters(model_param)
        model.create_model()
                     
        
        control_callback = model_utils.FitControl(mode = 'max', multiinput = False, verbose=self.verbose)
        
        model.model.fit( X_train, y_train, validation_data = (X_test, y_test), 
                                           epochs = max_epochs, batch_size = batch_size, callbacks = [control_callback],
                                           class_weight = class_weights, verbose = self.verbose)            
        
        score = control_callback.best_score
        
        del model
        K.clear_session() 
        
        return score
    
    
    def kfold_val(self, X, y=None, model_param = None, kfold_splits = 5, kfold_function=KFold, 
                  max_epochs=20, batch_size=100, seed=42, verbose=0, early_stopping=True):

        mem = psutil.virtual_memory()
        print( 'Memory:', mem[2] )
        
        cvscores = []    
        
        enum = enumerate(kfold_function(n_splits=kfold_splits, shuffle=True, random_state=seed).split(X,y))
        for i,(index_train, index_valid) in enum:
            
            if self.train_on_batch == False:
                X_train, X_test = X[ index_train ], X[ index_valid ]
                y_train, y_test = y[ index_train ], y[ index_valid ]

                
                score = self.train( X_train, y_train, X_test, y_test, max_epochs = max_epochs, model_param = model_param,
                                                         batch_size = batch_size, early_stopping = True)
                
            else:
                
                X_train, X_test = X[ index_train ], X[ index_valid ]
                score = self.train_in_batches( X_train, X_test, max_epochs = max_epochs, model_param = model_param,
                                                         batch_size_train = batch_size, early_stopping = True)
            
            cvscores.append(score)
            
        return cvscores

  

    def find_optimal_parameters( self, X, y=None, cv=False, kfold_splits=3, test_size = 0.33, num_calls=12,
                                 max_epochs=100, batch_size=2, seed=42, verbose=1): 

        mem = psutil.virtual_memory()
        print( 'Memory:', mem[2] )

        # Initialize the skopt params
        model = self.get_model()
        dimensions = model.get_skopt_dimensions()
        
        prior_values = []
        prior_names = []
        for var in dimensions:
            name = var.name
            print( name )
            prior_names.append(name)
            prior_values.append(model.hp[name])
        
        
        if self.train_on_batch == False:
            prior_values.append( batch_size )
            prior_names.append( 'batch_size' )
        
        del model
        K.clear_session() 
                                
        global num_skopt_call
        num_skopt_call = 0
        cv_results = []

        @use_named_args(dimensions)
        def fitness(**p): 

            global num_skopt_call

            if verbose != 0: print('\n \t ::: {} SKOPT CALL ::: \n'.format(num_skopt_call+1))
            print(p)
            
            
            if cv == True:
                if self.train_on_batch == False:
                    batch_size_optimize = p.pop('batch_size', None)
                    cvscores = self.kfold_val(X, y, model_param = p, kfold_splits=kfold_splits, 
                                              max_epochs = max_epochs, batch_size = batch_size_optimize,
                                              early_stopping = True)
                else:
                    cvscores = self.kfold_val(X, model_param = p, kfold_splits=kfold_splits, 
                                              max_epochs = max_epochs, batch_size = batch_size,
                                              early_stopping = True)
                    
                result = -1 * np.mean( cvscores )
                std_dv = np.std( cvscores )
                print( result, std_dv )
                cv_results.append((num_skopt_call, result, std_dv))
                
            else:
                if self.train_on_batch == False:
                    batch_size_optimize = p.pop('batch_size', None)
                    X_train, X_test, y_train, y_test = self.split(X, y, test_size = test_size) 
                    score = self.train(  X_train, y_train, X_test, y_test, model_param = p, max_epochs = max_epochs, 
                                                batch_size = batch_size_optimize, early_stopping = True)
                else:
                    #batch_size = batch_size
                    X_train, X_test = self.split(X, test_size = test_size)              
                    score = self.train_in_batches( X_train, X_test, batch_size, model_param = p, max_epochs = max_epochs, 
                                                   early_stopping = True)               
                    
                result = -1 * score
                print( result )
                
            num_skopt_call += 1

            mem = psutil.virtual_memory()
            print( 'Memory:', mem[2] )

            return result

        search_result = gp_minimize( func = fitness, dimensions = dimensions,
                                     acq_func = 'EI', # Expected Improvement
                                     n_calls = num_calls, x0 = prior_values )

        if cv == True:
            result = self.store_results(search_result, prior_names, cv_results)
        else:
            result = self.store_results(search_result, prior_names)
        
        return result
    
    
    def store_results(self, search_result, prior_names, cv_results = None, store = True):

        params = pd.DataFrame(search_result['x_iters'])
        params.columns = [*prior_names]
        params = params.rename_axis('call').reset_index()
        scores = pd.DataFrame(search_result['func_vals'])
        scores.columns = ['cv_score']
        result = pd.concat([params, scores], axis=1)
        result = result.sort_values(by=['cv_score'])
        result.cv_score = result.cv_score * -1

        if cv_results is not None:
            cv_frame = pd.DataFrame(cv_results, columns = ['call', 'score', 'std'])
            cv_frame = cv_frame.drop(['score'], axis=1)
            result = pd.merge(result, cv_frame, on=['call'])

        if self.store == True:
            result.to_hdf(self.path + '/' + 'skopt.h5', 'frame')

        return result 
    
    
    
    

