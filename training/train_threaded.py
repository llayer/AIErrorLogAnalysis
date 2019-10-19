from threaded_skopt import dummy_func
import json
from threading import Thread
import os
import sys
import time
import psutil
import fit_handler
import warnings
import pandas as pd
import numpy as np
import setGPU
# Suppress warnings
warnings.filterwarnings("ignore")
import experiments as exp


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
    i_exp = int(o.pop('i_exp'))
    
    
    # Memory before the training
    mem = psutil.virtual_memory()
    print( 'Memory:', mem[2] )
    
    
    # Experiment parameters
    e = exp.experiments[ i_exp ]
    out_path = e.outpath + e.name + '/'
    
    # Load the data
    if 'NOMINAL' in e.name:
        path = e.inpath + 'input_' + 'NOMINAL' + '.h5'
        e.nlp_param['embedding_matrix_path'] = e.inpath + 'embedding_matrix_' + 'NOMINAL' + '.npy'
    elif 'AVG' in e.name:
        path = e.inpath + 'input_' + 'NOMINAL' + '.h5'
        e.nlp_param['embedding_matrix_path'] = e.inpath + 'embedding_matrix_' + 'NOMINAL' + '.npy'        
    else:
        path = e.inpath + 'input_' + 'VAR_DIM' + '.h5'
        e.nlp_param['embedding_matrix_path'] = e.inpath + 'embedding_matrix_' + 'VAR_DIM' + '.npy'
        
    
    actionshist, codes, sites = fit_handler.load_data(path)  
    
    # Setup the fit handler
    handler = fit_handler.FitHandler( exp.MODEL, codes, sites, exp.MAX_WORDS, 
                                     exp.GEN_PARAM, pruning_mode = exp.PRUNING,
                                     model_args = e['NLP_PARAM'], callback_args = e['CALLBACK'],
                                     train_on_batch = exp.TRAIN_ON_BATCH, verbose=2 )
    
    # Initial hyper parameters
    model_param = e.hyperparam
    # Overwrite with bayesian suggestion
    for name, value in o.items():
        model_param[name] = value

    """    
    score = handler.run_training(actionshist, batch_size = exp.BATCH_SIZE, max_epochs = exp.MAX_EPOCHS, 
                                     model_param = model_param)
    #value = -1 * score
    #print( value )
    
    """
    
    cvscores = handler.kfold_val( actionshist, model_param = model_param, kfold_splits = e.folds,
                                  max_epochs = e.max_epochs, batch_size = e.batch_size)
    

    value = -1 * np.mean( cvscores )
    std_dv = np.std( cvscores )
    
    res = {
        'result': value,
        'params' : o,
        'annotate' : 'a free comment',
        'std_dv': std_dv
    }
    print( res )
    if fold is not None:
        res['fold'] = fold
    out = out_path+hash_value
    dest = '%s.json'%out if fold is None else '%s_f%d.json'%(out, fold)
    
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
        
        #opt = {'learning_rate': 0.0006903190575459679, 'hash':111}
        print( opt )
        evaluate( opt )
                                                                                        
