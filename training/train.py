import setGPU
import psutil
import fit_handler
import warnings
import experiments as exp

# Suppress warnings
warnings.filterwarnings("ignore")

def train( i_exp = 0, model_param = None ):
    
    # Memory before the training
    mem = psutil.virtual_memory()
    print( 'Memory:', mem[2] )
    
    # Experiment parameters
    e = exp.EXPERIMENTS[ i_exp ]
    
    # Load the data
    path = exp.PATH + 'input_' + e['NAME'] + '.h5'
    actionshist, codes, sites = fit_handler.load_data(path, msg_only=exp.MSG_ONLY,
                                                      sample=exp.SAMPLE, sample_fact = exp.SAMPLE_FACT)
    
    # Setup the fit handler
    handler = fit_handler.FitHandler( exp.MODEL, codes, sites, exp.MAX_WORDS, exp.GEN_PARAM, nlp_param = e['NLP_PARAM'],
                                       train_on_batch = exp.TRAIN_ON_BATCH )

    if model_param is None:
        score = handler.run_training(actionshist, batch_size = exp.BATCH_SIZE, max_epochs = exp.MAX_EPOCHS, 
                                     model_param = e['HYPERPARAM'])
    else:
        score = handler.run_training(actionshist, batch_size = exp.BATCH_SIZE, max_epochs = exp.MAX_EPOCHS, 
                                     model_param = model_param)
    
    return score


if __name__ == "__main__":
    
    print( "Start training" )
    train()






