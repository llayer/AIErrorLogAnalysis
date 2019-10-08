from skopt.space import Real, Categorical, Integer

#Path to the input files
INPATH = '/nfshome/llayer/AIErrorLogAnalysis/data/'
# Test the fast I/O
#INPATH = '/imdata/error_log_analysis/data/'
OUTPATH = '/nfshome/llayer/AIErrorLogAnalysis/experiments_baseline/'

FOLDS = 3
SKOPTCALLS = 30
CV = True

# Epochs
MAX_EPOCHS = 200
BATCH_SIZE = 2000
MAX_WORDS = 0

# Skopt dimensions
SKOPT_DIM = [       
    Real(        low=1e-4, high=1e-1, prior='log-uniform', name='learning_rate'     ),
    Integer(     low=10,    high=100,                        name='dense_units'     ),
    Integer(     low=2,    high=8,                        name='dense_layers'       ),
    Real(        low=1e-5, high=0.9,  prior="log-uniform", name='regulizer_value'   ),
    Real(        low=0.0, high=0.5,                       name='dropout_value'     ),
    Integer(     low=500,   high = 5000,                    name='batch_size'       )
]

# Callback
cb = { 'es': True, 'patience': 10, 'kill_slowstarts': False, 'kill_threshold': 0.5001, 'store_best_roc': False }
# Initial param
hp = {'learning_rate':0.005675, 'dense_units':35, 'dense_layers' : 6, 'regulizer_value' : 0.001000, 'dropout_value' : 0.052315 }

# sample
SAMPLE = False
SAMPLE_FACT = 5

# Model
MODEL = 'baseline'

# Defines the input experiments for the machine learning
EXPERIMENTS = [
    
    # 1st experiment 
    {'NAME': 'BASELINE', 'PRUNE': 'None',
     'HYPERPARAM': hp, 'CALLBACK': cb} ,
    
    # 2nd experiment: prune the negative only sites
    {'NAME': 'BASELINE_PRUNE_NEG', 'PRUNE': 'Neg',
     'HYPERPARAM': hp, 'CALLBACK': cb } ,
    
    # 1st experiment initial parameter
    {'NAME': 'BASELINE_PRUNE_TIERS', 'PRUNE': 'Tiers',
     'HYPERPARAM': hp, 'CALLBACK': cb } ,
    
    # 1st experiment initial parameter
    {'NAME': 'BASELINE_PRUNE_FREQ', 'PRUNE':'LowFreq',
     'HYPERPARAM': hp, 'CALLBACK': cb } 
    
]
