import os
import shutil
import setGPU
import psutil
import fit_handler
import warnings
import baseline_experiments as exp

# Suppress warnings
warnings.filterwarnings("ignore")


def create_dir(path, overwrite):

    if overwrite == True:
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)
        print( 'Create directory:', path )
    else:
        if os.path.exists(path):    
            print( 'Directory already exists! If you want to remove it do:' )
            print( 'rm -r', path ) 
        else:
            os.makedirs(path)
            print( 'Create directory:', path )


def train( i_exp = 0, mode = 'train' ):
    
    # Memory before the training
    mem = psutil.virtual_memory()
    print( 'Memory:', mem[2] )
    
    # Experiment parameters
    e = exp.EXPERIMENTS[ i_exp ]
    
    # Store path
    overwrite = False
    outpath = exp.OUTPATH + e['NAME'] + '/'
    create_dir(outpath, overwrite)
    
    # Load the data
    inpath = exp.INPATH + 'input_NOMINAL' + '.h5'
    actionshist, codes, sites = fit_handler.load_data(inpath)
    
    # Setup the fit handler
    handler = fit_handler.FitHandler( exp.MODEL, codes, sites, pruning_mode = e['PRUNE'],
                                      callback_args = e['CALLBACK'], train_on_batch = False, path = outpath, verbose=2)
    
    # Get the count matrix
    X,y = handler.count_matrix(actionshist)
    
    mem = psutil.virtual_memory()
    print( 'Memory:', mem[2] )
    
    if mode == 'optimize':
        score = handler.find_optimal_parameters( exp.SKOPT_DIM, e['HYPERPARAM'], X, y, cv=exp.CV, kfold_splits=exp.FOLDS, 
                                                num_calls=exp.SKOPTCALLS, max_epochs=exp.MAX_EPOCHS, batch_size=exp.BATCH_SIZE)
    elif mode == 'cv':
        score = handler.kfold_val( X, y, model_param = e['HYPERPARAM'], kfold_splits = exp.FOLDS,
                                   max_epochs = exp.MAX_EPOCHS, batch_size = exp.BATCH_SIZE)
    else:
        score = handler.run_training(X, y, batch_size = exp.BATCH_SIZE, max_epochs = exp.MAX_EPOCHS, 
                                     model_param = e['HYPERPARAM'])    
        
        

    
    return score


if __name__ == "__main__":
    
    print( "Start training" )
    train(i_exp = 1, mode = 'train')
