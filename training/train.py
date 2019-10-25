import setGPU
import psutil
import fit_handler
import warnings
import numpy as np
import experiments as exp

# Suppress warnings
warnings.filterwarnings("ignore")

def train( i_exp = 0, model_param = None, sample_frac = None ):
    
    # Memory before the training
    mem = psutil.virtual_memory()
    print( 'Memory:', mem[2] )
    
    # Experiment parameters
    e = exp.experiments[ i_exp ]
    
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
        
        
    if sample_frac == None:
        actionshist, codes, sites = fit_handler.load_data(path)
    else:
        actionshist, codes, sites = fit_handler.load_data(path, sample = True, sample_frac = sample_frac)
    
    # Setup the fit handler
    handler = fit_handler.FitHandler( e.model, codes, sites, e.max_words, 
                                     e.gen_param, pruning_mode = e.pruning,
                                     model_args = e.nlp_param, callback_args = e.callback,
                                     train_on_batch = e.train_on_batch, verbose=2 )

    if model_param is None:
        model_param = e.hyperparam
    #score = handler.run_training(actionshist, batch_size = exp.BATCH_SIZE, max_epochs = exp.MAX_EPOCHS, 
    #                             model_param = e['HYPERPARAM'])
    cvscores = handler.kfold_val( actionshist, model_param = model_param, kfold_splits = e.folds,
               max_epochs = e.max_epochs, batch_size = e.batch_size)
    


if __name__ == "__main__":
    
    print( "Start training" )
    train(i_exp = 4 , sample_frac = 0.125)
    #fractions = np.linspace(0.1, 1, 5, endpoint=True)
    #for f in fractions:
    #    train(i_exp = 1, sample_frac = f)





