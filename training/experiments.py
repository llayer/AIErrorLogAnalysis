import keras
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from skopt.space import Real, Categorical, Integer

#Path to the input files
INPATH = '/nfshome/llayer/AIErrorLogAnalysis/data/'
# Test the fast I/O
#INPATH = '/imdata/error_log_analysis/data/'
OUTPATH = '/nfshome/llayer/AIErrorLogAnalysis/experiments/'

# Include counts
MSG_ONLY = False
PRUNING = 'Neg'

# Skopt dimensions
SKOPT_DIM = [
    Real(        low=1e-5, high=1e-3, prior='log-uniform', name='learning_rate'     ),
    Real(        low=1e-3, high=0.1, prior='log-uniform', name='dropout'     ),
    #Real(        low=1e-4, high=0.9,  prior="log-uniform", name='l2_regulizer'   ),
    Integer(     low=2, high=20,                          name='rnn_units'   ),
    #Integer(     low=2, high = 20,                       name = 'units_site'    ),
    Integer(     low=1,    high=3,                         name='dense_layers'      ),
    Integer(     low=10,    high=50,                         name='dense_units'      ),
    #Integer(     low=0,    high=1,                         name='encode_sites'      ),
    Integer(     low=0,    high=1,                         name='train_embedding'      ),
    ]

# batch_size and epochs 
BATCH_SIZE = 1
MAX_EPOCHS = 20

# sample
SAMPLE = False
SAMPLE_FACT = 5

# batch generator param
AVG_W2V = False
MAX_WORDS = 400
GEN_PARAM = {}
GEN_PARAM['averaged'] = AVG_W2V
GEN_PARAM['only_msg'] = MSG_ONLY 
GEN_PARAM['sequence'] = False
GEN_PARAM['max_msg'] = 1
GEN_PARAM['cut_front'] = True
TRAIN_ON_BATCH = True

# Model
MODEL = 'nlp_msg'

# Defines the input experiments for the machine learning
EXPERIMENTS = [
    
    # 1st experiment initial parameter
    {'NAME': 'NOMINAL_t', 'DIM':50, 'VOCAB': -1, 'ALGO': 'sg',
     'NLP_PARAM': {'cudnn': False, 'batch_norm': False, 'word_encoder': 'LSTM', 
                   'attention': False, 'include_counts': True, 'avg_w2v': AVG_W2V},
     'HYPERPARAM': { 'dropout':0.0, 'rec_dropout':0.0, 'rnn': GRU, 'rnn_units' : 20, 'activation_site': 'relu', 
                    'l2_regulizer': 0.0001, 'encode_sites': False, 'units_site': 10, 'dense_layers': 5, 
                    'train_embedding': True, 'dense_units': 50, 'learning_rate':0.0001 } ,
     'CALLBACK': { 'es': True, 'patience': 3, 'kill_slowstarts': True, 'kill_threshold': 0.51 }
     
    } ,
    
    # 2nd experiment averaged
    {'NAME': 'AVG', 'DIM':50, 'VOCAB': -1, 'ALGO': 'sg',
     'NLP_PARAM': {'cudnn': False, 'batch_norm': False, 'word_encoder': 'LSTM', 
                   'attention': False, 'include_counts': True, 'avg_w2v': AVG_W2V},
     'HYPERPARAM': { 'dropout':0.0, 'rec_dropout':0.0, 'rnn': GRU, 'rnn_units' : 10, 'activation_site': 'relu', 
                    'l2_regulizer': 0.0001, 'encode_sites': False, 'units_site': 10, 'dense_layers': 3, 
                    'train_embedding': True, 'dense_units': 20, 'learning_rate':0.0001 } ,
     'CALLBACK': { 'es': True, 'patience': 3, 'kill_slowstarts': True, 'kill_threshold': 0.5001 }
     
    } ,
    
    """
    # 2nd experiment - train the embeddings
    {'NAME': 'TRAIN_EMBEDDINGS', 'DIM':50, 'VOCAB': -1, 'ALGO': 'sg' ,
     'NLP_PARAM': { 'cudnn': False, 'batch_norm': False, 'train_embedding': True, 'word_encoder': 'LSTM', 
                   'attention': False, 'encode_sites': True, 'include_counts': False},
     'HYPERPARAM': { 'dropout':0.0, 'rec_dropout':0.0, 'rnn': GRU, 'rnn_units' : 10, 'activation_site': 'relu', 
                    'units_site': 10, 'dense_layers': 3, 'dense_units': 20, 'learning_rate':0.0001 } ,
     
    } ,
    
    # 3rd experiment - no site encoding
    {'NAME': 'NO_SITE_ENCODING', 'DIM':50, 'VOCAB': -1, 'ALGO': 'sg' ,
     'NLP_PARAM': { 'cudnn': False, 'batch_norm': False, 'train_embedding': False, 'word_encoder': 'LSTM', 
                   'attention': False, 'encode_sites': False, 'include_counts': False},
     'HYPERPARAM': { 'dropout':0.0, 'rec_dropout':0.0, 'rnn': GRU, 'rnn_units' : 10, 'activation_site': 'relu', 
                    'units_site': 10, 'dense_layers': 3, 'dense_units': 20, 'learning_rate':0.0001 } ,
     
    } ,
     
    # 4th experiment - Conv1D instead of GRU
    
    {'NAME': 'CONV_1D', 'DIM':50, 'VOCAB': -1, 'ALGO': 'sg'
     'NLP_PARAM': { 'cudnn': False, 'batch_norm': False, 'train_embedding': False, 'word_encoder': 'LSTM', 
                   'attention': False, 'encode_sites': True, 'include_counts': False},
     'HYPERPARAM': { 'dropout':0.0, 'rec_dropout':0.0, 'rnn': GRU, 'rnn_units' : 10, 'activation_site': 'relu', 
                    'units_site': 10, 'dense_layers': 3, 'dense_units': 20, 'learning_rate':0.0001 } ,
     
    } ,
    """
    
    # Decay test: decay = lr / total number of epochs
]
