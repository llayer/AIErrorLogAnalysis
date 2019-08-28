from threaded_skopt import dummy_func
import json
from threading import Thread
import os
import sys
import time
import keras
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
import psutil
import fit_handler
import warnings
import pandas as pd
import numpy as np
import setGPU
# Suppress warnings
warnings.filterwarnings("ignore")

class fold(Thread):
    def __init__(self, opt, fold):
        Thread.__init__(self)
        self.opt = opt
        self.fold = fold
    def run(self):
        command = 'python '+' '.join(sys.argv)+' --fold %d'%self.fold
        print( "running",command )
        os.system( command )
    
def evaluate( o , fold = None):
    
    hash_value = o.pop('hash')
    
    #X = (o['par0'], o['par1'])
    # Memory before the training
    mem = psutil.virtual_memory()
    print( 'Memory:', mem[2] )

    # Load the data
    path = '/nfshome/llayer/data/input_msg.h5'
    actionshist = pd.read_hdf(path, 'frame', start = 0, stop = 1000)
    #sites = pd.read_hdf(path, 'frame2')
    #codes = pd.read_hdf(path, 'frame3')
    codes = pd.read_hdf(path, 'frame4')
    sites = pd.read_hdf(path, 'frame5')
    codes.rename({'errors_msg': 'error'}, axis=1, inplace=True)
    sites.rename({'sites_msg': 'site'}, axis=1, inplace=True)


    nlp_param = {
        'cudnn': True, 'batch_norm': False, 'train_embedding': False, 'word_encoder': 'LSTM', 'attention': False,
        'encode_sites': False, 'include_counts': False
    }
    
    # Setup the fit handler
    embedding_dim = 300
    gen_param = {}
    gen_param['averaged'] = False
    gen_param['only_msg'] = True
    gen_param['first_only'] = True
    gen_param['max_msg'] = 5

    handler = fit_handler.FitHandler( 'nlp_msg', codes, sites, embedding_dim, gen_param, nlp_param = nlp_param,
                                     train_on_batch = True )

    value = -1 * handler.run_training(actionshist, batch_size = 2, max_epochs = 2, model_param = o)
    print( value )
    #dummy_func( X , fold = fold)
    res = {
        'result': value,
        'params' : o,
        'annotate' : 'a free comment'
    }
    print( res )
    if fold is not None:
        res['fold'] = fold
    dest = '%s.json'%hash_value if fold is None else '%s_f%d.json'%(hash_value, fold)
    
    open(dest,'w').write(json.dumps(res))

def evaluate_folds( o , Nfolds , Nthreads=2):
    ## thats a dummy way of running folds sequentially
    #for f in range(Nfolds):
    #    evaluate( opt, fold = f )

    folds = []
    for f in range(Nfolds):
        folds.append( fold( opt, fold = f ) )

    ons = []
    while True:
        if len(ons) < Nthreads:
            ons.append( folds.pop(-1) )
            ons[-1].start()
            time.sleep(2)
            continue
        for f in ons:
            if not f.is_alive():
                ons.remove( f )
                break
        if len(folds) == 0:
            break
        
    ## read all back and make the average
    r = []
    for f in range(Nfolds):
        src = '%s_f%d.json'%(o['hash'], f)
        d = json.loads( open(src).read())
        r.append( d['result'] )
    import numpy as np

    ## that's the final expect answer
    dest = '%s.json'%o['hash']
    res = {
        'result': np.mean( r ),
        'params' : o,
    }
    print( "The averaged value on hash",o['hash'],"is",res )
    open(dest,'w').write(json.dumps(res))
    
        
if __name__ == "__main__":
    import sys
    ## convert blindy the parameters
    opt={}
    for i,_ in enumerate(sys.argv):
        k = sys.argv[i]
        if k.startswith('--'):
            v = sys.argv[i+1]
            try:
                opt[k[2:]] = float(v)
            except:
                opt[k[2:]] = v
    Nfolds = int(opt.pop('folds')) if 'folds' in opt else 1
    if 'fold' in opt:
        f = int(opt.pop('fold'))
        evaluate( opt, fold = f )
    elif Nfolds>1:
        ## need to spawn enough threads, and collect them all
        print( "going for",Nfolds,"folds" )
        evaluate_folds( opt, Nfolds = Nfolds )
    else:
        
        """
        opt = {
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
        """
        #opt = {'learning_rate': 0.0006903190575459679, 'hash':111}
        #print( opt )
        evaluate( opt )
                                                                                        
