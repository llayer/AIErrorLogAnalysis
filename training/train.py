import pandas as pd
import numpy as np
import setGPU
import psutil
import importlib
import fit_handler
import warnings
import psutil
import keras
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU

# Suppress warnings
warnings.filterwarnings("ignore")

def train( model_param = None ):
    
    # Memory before the training
    mem = psutil.virtual_memory()
    print( 'Memory:', mem[2] )

    # Load the data
    path = '/nfshome/llayer/data/input_msg.h5'
    actionshist = pd.read_hdf(path, 'frame', start=0, stop=1000)
    #sites = pd.read_hdf(path, 'frame2')
    #codes = pd.read_hdf(path, 'frame3')
    codes = pd.read_hdf(path, 'frame4')
    sites = pd.read_hdf(path, 'frame5')
    codes.rename({'errors_msg': 'error'}, axis=1, inplace=True)
    sites.rename({'sites_msg': 'site'}, axis=1, inplace=True)

    
    hyperparam = {
    # Regularization
    'l2_regulizer': keras.regularizers.l2(0.0001),
    'dropout':0.0,
    # Conv1D
    'conv_layers':1,
    'max_pooling':3,
    'filters':64,
    'kernel_size':3,
    'units_conv':10,
    # RNN with optional attention
    'att_units':10,
    'rec_dropout':0.0,
    'rnn': GRU, #TRY LSTM
    'rnncud': CuDNNLSTM, # TRY  CuDNNGRU
    'rnn_units' : 10,
    # Site encoding
    'activation_site': 'relu', #TRY linear
    'units_site': 10,
    # Final layers
    'dense_layers': 3,
    'dense_units': 20,
    'learning_rate':0.0001
            }

    nlp_param = {
        'cudnn': True, 'batch_norm': False, 'train_embedding': False, 'word_encoder': 'Conv1D', 'attention': True,
        'encode_sites': False, 'include_counts': False
    }
    
    # Setup the fit handler
    embedding_dim = 500
    gen_param = {}
    gen_param['averaged'] = False
    gen_param['only_msg'] = True
    gen_param['first_only'] = True
    gen_param['max_msg'] = 5

    handler = fit_handler.FitHandler( 'nlp_msg', codes, sites, embedding_dim, gen_param, nlp_param = nlp_param,
                                     train_on_batch = True )

    if model_param is None:
        score = handler.run_training(actionshist, batch_size = 2, max_epochs = 2, model_param = hyperparam)
    else:
        score = handler.run_training(actionshist, batch_size = 2, max_epochs = 2, model_param = model_param)

    # Train the baseline
    #X,y = handler.count_matrix(actionshist)
    #handler.test_training(X, y, batch_size = 100, max_epochs = 2)

    # Bayesian search
    #result = handler.find_optimal_parameters( actionshist, cv=False, num_calls=20, max_epochs=20, batch_size=100)
    
    return score

if __name__ == "__main__":
    
    print( "Start training" )
    train()






