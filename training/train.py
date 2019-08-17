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
#codes_msg = pd.read_hdf(path, 'frame4')
#sites_msg = pd.read_hdf(path, 'frame5')
#codes_msg.rename({'errors_msg': 'errors'}, axis=1, inplace=True)
#sites_msg.rename({'sites_msg': 'sites'}, axis=1, inplace=True)

# Setup the fit handler
embedding_dim = 50
gen_param = {}
gen_param['averaged'] = True
gen_param['only_msg'] = True
gen_param['first_only'] = True
gen_param['max_msg'] = 5
handler = fit_handler.FitHandler( 'baseline', codes, sites, embedding_dim, gen_param, train_on_batch = False, verbose=2 )

# Train the baseline
X,y = handler.count_matrix(actionshist)
handler.test_training(X, y, batch_size = 100, max_epochs = 2)



