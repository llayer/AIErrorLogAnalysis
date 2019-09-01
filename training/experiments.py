import keras
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU

#Path to the input files
INPATH = '/nfshome/llayer/AIErrorLogAnalysis/data/'
OUTPATH = '/nfshome/llayer/AIErrorLogAnalysis/experiments/'

# batch_size and epochs 
BATCH_SIZE = 2
MAX_EPOCHS = 10

# sample
SAMPLE = True
SAMPLE_FACT = 5

# batch generator param
MAX_WORDS = 400
GEN_PARAM = {}
GEN_PARAM['averaged'] = False
GEN_PARAM['only_msg'] = True
GEN_PARAM['sequence'] = False
GEN_PARAM['max_msg'] = 5
GEN_PARAM['cut_front'] = True
MSG_ONLY = True
TRAIN_ON_BATCH = True

# Model
MODEL = 'nlp_msg'

# Defines the input experiments for the machine learning
EXPERIMENTS = [
    
    # 1st experiment 
    {'NAME': 'NOMINAL', 'DIM':50, 'VOCAB': -1, 'ALGO': 'sg',
     'NLP_PARAM': { 'cudnn': False, 'batch_norm': False, 'train_embedding': False, 'word_encoder': 'LSTM', 
                   'attention': False, 'encode_sites': True, 'include_counts': False},
     'HYPERPARAM': { 'dropout':0.0, 'rec_dropout':0.0, 'rnn': GRU, 'rnn_units' : 10, 'activation_site': 'relu', 
                    'units_site': 10, 'dense_layers': 3, 'dense_units': 20, 'learning_rate':0.0001 } ,
     
    } ,
    
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
    """
    {'NAME': 'CONV_1D', 'DIM':50, 'VOCAB': -1, 'ALGO': 'sg'
     'NLP_PARAM': { 'cudnn': False, 'batch_norm': False, 'train_embedding': False, 'word_encoder': 'LSTM', 
                   'attention': False, 'encode_sites': True, 'include_counts': False},
     'HYPERPARAM': { 'dropout':0.0, 'rec_dropout':0.0, 'rnn': GRU, 'rnn_units' : 10, 'activation_site': 'relu', 
                    'units_site': 10, 'dense_layers': 3, 'dense_units': 20, 'learning_rate':0.0001 } ,
     
    } ,
    """
]