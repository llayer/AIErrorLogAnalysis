import keras
import pandas as pd
import numpy as np
import psutil
import setGPU
from models import baseline_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
#from models import nlp_model
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
from scipy import interp

# Early stopping
class EarlyStoppingRoc(keras.callbacks.Callback):

    def __init__(self, patience, mode='min'):

        super().__init__()
        self.patience = patience
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.wait = 0
        self.is_better = np.greater if mode == 'max' else np.less
        self.reports = []


    def on_epoch_end(self, epoch, logs={}):

        val_predict = (np.asarray(self.model.predict(self.validation_data[0])))
        val_predict_max = np.argmax(val_predict, axis=-1)
        val_targ = self.validation_data[1]
        precision, recall, thresholds = precision_recall_curve(val_targ, val_predict)
        score = roc_auc_score(val_targ, val_predict)
        monit = (auc(recall, precision), average_precision_score(val_targ, val_predict))
        self.reports.append(score)
        #print(val_predict_max)
        print( len(val_predict_max[val_predict_max==1]) )
        print
        print( score )
        print( monit )
        print
        if self.is_better(score, self.best_score):
            self.wait = 0
            self.best_score = score
        else:
            self.wait += 1
        if self.wait > self.patience:
            self.model.stop_training = True



class FitHandler(object):
    

    def __init__(self):
        pass
        

    def train(self, p, X_train, y_train, X_test, y_test, batch_size = 100, max_epochs = 200, early_stopping = True):

        dim_errors = X_train.shape[1]
        dim_sites = X_train.shape[2]

        class_weights = class_weight.compute_class_weight('balanced', np.unique( y_train), y_train)

        model_optimize = baseline_model.FF(2, dim_errors, dim_sites)
        model_optimize.create_model( p['learning_rate'], p['dense_units'], 
                             p['dense_layers'], p['regulizer_value'], p['dropout_value'] )  
        
        es = EarlyStoppingRoc(mode='max', patience=5)

        model_optimize.model.fit( X_train, y_train, validation_data = (X_test, y_test), 
                                           epochs = max_epochs, batch_size = batch_size, callbacks =[es],
                                           class_weight=class_weights)            
        
        y_pred = model_optimize.predict(X_test, argmax=False)
        
        del model_optimize
        K.clear_session() 
        
        print(psutil.cpu_percent(), psutil.virtual_memory())

        return y_test, y_pred, es
    
    
    def kfold_val(self, p, X, y, kfold_splits = 5, kfold_function=KFold, 
                  max_epochs=100, batch_size=256, seed=42, verbose=0, early_stopping=False, optimize = True):

        print(psutil.cpu_percent(), psutil.virtual_memory())
        
        cvscores = []
        early_stoppings = []
        tprs = []
        fprs = []
        aucs = []        
        
        enum = enumerate(kfold_function(n_splits=kfold_splits, shuffle=True, random_state=seed).split(X,y))
        for i,(index_train, index_valid) in enum:
            
            
            X_train, X_test = X[ index_train ], X[ index_valid ]
            y_train, y_test = y[ index_train ], y[ index_valid ]

            es = EarlyStoppingRoc(mode='max', patience=3)
    
            
            y_test, y_pred, es = self.train( p, X_train, y_train, X_test, y_test, max_epochs = max_epochs, 
                                                     batch_size = batch_size, early_stopping = True)
            
            if optimize == True:
                score = roc_auc_score(y_test, y_pred)
                cvscores.append(score)
                
            else:
                early_stoppings.append(es)
                # Compute ROC curve and area the curve
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                fprs.append(fpr)
                tprs.append(tpr)
                aucs.append(roc_auc)
                

        if optimize == True:
            return cvscores
        else:
            return {'es': early_stoppings, 'aucs': aucs, 'tprs': tprs, 'fprs': fprs}
    
  

    def find_optimal_parameters( self, model, X, y, cv=True, kfold_splits=3, num_calls=12,
                                 max_epochs=100, batch_size=100, seed=42, verbose=1 ): 

        print(psutil.cpu_percent(), psutil.virtual_memory())

        global num_skopt_call
        num_skopt_call = 0
        dimensions = model.get_skopt_dimensions()
        prior_values = [a for a in model.model_params.values()]
        prior_values.append( batch_size )
        prior_names = [a for a in model.model_params.keys()]
        prior_names.append( 'batch_size' )
        
        cv_results = []

        @use_named_args(dimensions)
        def fitness(**p): # p = { 'p1':0.1,'p2':3,... }
            #global y_test
            #y_test = y_t
            global num_skopt_call

            if verbose != 0: print('\n \t ::: {} SKOPT CALL ::: \n'.format(num_skopt_call+1))
            print(p)
            
            
            #class_weights = class_weight.compute_class_weight('balanced', np.unique( y_train), y_train)

            if cv == True:
                cvscores = self.kfold_val(p, X, y, kfold_splits=kfold_splits, 
                                          max_epochs = max_epochs, batch_size = p['batch_size'],
                                          early_stopping = True)
                result = -1 * np.mean( cvscores )
                std_dv = np.std( cvscores )
                print( result, std_dv )
                
                cv_results.append((num_skopt_call, result, std_dv))
                
            else:
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)              
                y_test, y_pred = self.train( p, X_train, y_train, X_test, y_test, max_epochs = 200, 
                                            batch_size = p['batch_size'], early_stopping = True)
                result = -1 * roc_auc_score(y_test, y_pred)
                print(psutil.cpu_percent(), psutil.virtual_memory())
                
         

            print( result )
            num_skopt_call += 1

            print(psutil.cpu_percent(), psutil.virtual_memory())

            return result

        search_result = gp_minimize( func = fitness, dimensions = dimensions,
                                     acq_func = 'EI', # Expected Improvement.
                                     n_calls = num_calls, x0 = prior_values )

        return search_result, cv_results
    
    
    def get_results(self, search_result, prior_names):

        params = pd.DataFrame(search_result['x_iters'])
        params.columns = [*prior_names]
        scores = pd.DataFrame(search_result['func_vals'])
        scores.columns = ['cv_score']
        res = pd.concat([params, scores], axis=1)
        res = res.sort_values(by=['cv_score'])
        return res    
    
    
    
    

