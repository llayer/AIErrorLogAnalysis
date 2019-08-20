import pandas as pd
import numpy as np
import setGPU
import psutil
import importlib
import fit_handler
import warnings
import psutil

# Suppress warnings
warnings.filterwarnings("ignore")

# Memory before the training
mem = psutil.virtual_memory()
print( 'Memory:', mem[2] )

# Load the data
path = '/nfshome/llayer/data/input_msg.h5'
actionshist = pd.read_hdf(path, 'frame')
sites = pd.read_hdf(path, 'frame2')
codes = pd.read_hdf(path, 'frame3')
#codes = pd.read_hdf(path, 'frame4')
#sites = pd.read_hdf(path, 'frame5')
#codes.rename({'errors_msg': 'error'}, axis=1, inplace=True)
#sites.rename({'sites_msg': 'site'}, axis=1, inplace=True)

# Setup the fit handler
embedding_dim = 50
gen_param = {}
gen_param['averaged'] = True
gen_param['only_msg'] = False
gen_param['first_only'] = True
gen_param['max_msg'] = 5
handler = fit_handler.FitHandler( 'nlp_w2v_counts', codes, sites, embedding_dim, gen_param, pruning_mode = 'Neg', name = 'test2',
                                 train_on_batch = True, verbose=2 )

# Train the baseline
#X,y = handler.count_matrix(actionshist)
#handler.test_training(X, y, batch_size = 100, max_epochs = 2)

# Bayesian search
result = handler.find_optimal_parameters( actionshist, cv=False, num_calls=20, max_epochs=20, batch_size=100)





