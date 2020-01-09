import keras
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from skopt.space import Real, Categorical, Integer

class BaseExperiment(object):
    
    def __init__(self, name, batch_size, max_epochs, max_words, avg_w2v = False, pruning = 'Neg', folds = 3, 
                 train_on_batch = True, msg_only = False, model = 'nlp_msg'):
        
        # Paths
        self.inpath = '/storage/user/llayer/AIErrorLogAnalysis/data/'
        self.outpath = '/storage/user/llayer/AIErrorLogAnalysis/experiments/'
        
        # Base variables
        self.name = name
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_words = max_words
        self.pruning = pruning
        self.folds = folds
        self.msg_only = msg_only
        self.train_on_batch = train_on_batch
        self.avg_w2v = avg_w2v
        self.model = model
        
        # batch generator param
        self.gen_param = {}
        self.gen_param['averaged'] = avg_w2v
        self.gen_param['only_msg'] = msg_only 
        self.gen_param['sequence'] = False
        self.gen_param['max_msg'] = 1
        self.gen_param['cut_front'] = True
        
        # Callback
        self.callback = { 'es': True, 'patience': 3, 'kill_slowstarts': False, 'kill_threshold': 0.51 }
    
    
    def set_nlp_param(self, attention = False, cudnn = False, batch_norm = False, word_encoder = 'LSTM', 
                      include_counts = True, init_embedding = True):
        
        self.nlp_param = {'cudnn': cudnn, 'batch_norm': batch_norm, 'word_encoder': word_encoder, 
                         'attention': attention, 'include_counts': include_counts, 
                         'avg_w2v' : self.avg_w2v, 'init_embedding': init_embedding}      
        
        
    def set_hp(self, dropout = 0.0, rec_dropout = 0.0, rnn = GRU, rnn_units = 20, activation_site = 'relu', 
                    l2_regulizer = 0.0001, encode_sites = False, units_site = 10, dense_layers = 5, 
                    train_embedding = True, dense_units = 50, embedding = 20, learning_rate = 0.0001):
        
        self.hyperparam = { 'dropout': dropout, 'rec_dropout':rec_dropout, 'rnn': rnn, 'rnn_units' : rnn_units,
                           'activation_site': activation_site, 'l2_regulizer': l2_regulizer, 'encode_sites': encode_sites, 
                           'units_site': units_site, 'dense_layers': dense_layers, 'train_embedding': train_embedding, 
                           'dense_units': dense_units, 'embedding': embedding, 'learning_rate': learning_rate }
        
    def set_skopt_dim(self, skopt_dim):
        
        self.skopt_dim = skopt_dim
               

    
    
# 1st experiment
nominal = BaseExperiment('NOMINAL', batch_size = 1, max_epochs = 15, max_words = 400)
# Skopt dimensions
skopt_dim_nominal = [
    Real(        low=1e-5, high=1e-3, prior='log-uniform', name='learning_rate'     ),
    Real(        low=1e-3, high=0.1, prior='log-uniform', name='dropout'     ),
    Real(        low=1e-4, high=0.9,  prior="log-uniform", name='l2_regulizer'   ),
    Integer(     low=5, high=50,                          name='rnn_units'   ),
    Integer(     low=1,    high=5,                         name='dense_layers'      ),
    Integer(     low=10,    high=50,                         name='dense_units'      ),
    ]
nominal.set_hp()
nominal.set_nlp_param()
nominal.set_skopt_dim(skopt_dim_nominal)

    
# 2nd experiment lower embedding
dim20 = BaseExperiment('VAR_DIM', batch_size = 1, max_epochs = 12, max_words = 400)
skopt_dim_20 = [
    Real(        low=1e-5, high=1e-3, prior='log-uniform', name='learning_rate'     ),
    Real(        low=1e-3, high=0.1, prior='log-uniform', name='dropout'     ),
    Integer(     low=2, high=20,                          name='rnn_units'   ),
    Integer(     low=1,    high=5,                         name='dense_layers'      ),
    Integer(     low=10,    high=50,                         name='dense_units'      ),
    Integer(     low=0,    high=1,                         name='train_embedding'      ),
    ]
dim20.set_hp()
dim20.set_nlp_param()
dim20.set_skopt_dim(skopt_dim_20)



# 3nd experiment averaged
avg = BaseExperiment('AVG', batch_size = 100, max_epochs = 200, max_words = 50, avg_w2v = True)
skopt_dim_avg = [
    Real(        low=1e-5, high=1e-3, prior='log-uniform', name='learning_rate'     ),
    Real(        low=1e-5, high=0.1, prior='log-uniform', name='dropout'     ),
    Real(        low=1e-5, high=0.9,  prior="log-uniform", name='l2_regulizer'   ),
    Integer(     low=2, high = 20,                       name = 'units_site'    ),
    #Integer(     low=2,    high=10,                         name='pool_size'      ),
    Integer(     low=1,    high=5,                         name='dense_layers'      ),
    Integer(     low=10,    high=100,                         name='dense_units'      ),
    ]
# CHECK Callback
avg.set_hp(dropout = 0.00001, l2_regulizer = 0.000042, units_site = 19.0, dense_layers = 3, 
                    dense_units = 30, learning_rate = 0.000027)
avg.set_nlp_param()
avg.set_skopt_dim(skopt_dim_avg)
 
    
# 4th experiment attention
dim20_att = BaseExperiment('VAR_DIM_Att', batch_size = 1, max_epochs = 15, max_words = 400)
skopt_dim_20_att = [
    Real(        low=1e-5, high=1e-3, prior='log-uniform', name='learning_rate'     ),
    Real(        low=1e-3, high=0.1, prior='log-uniform', name='dropout'     ),
    Integer(     low=2, high=20,                          name='rnn_units'   ),
    Integer(     low=1,    high=5,                         name='dense_layers'      ),
    Integer(     low=10,    high=50,                         name='dense_units'      ),
    Integer(     low=2,    high=20,                         name='att_units'      ),
    ]
dim20_att.set_hp()
dim20_att.set_nlp_param(attention = True)
dim20_att.set_skopt_dim(skopt_dim_20)
    
    
# 5th experiment embedding varied
embedding = BaseExperiment('EMBEDDING', batch_size = 1, max_epochs = 12, max_words = 400)
skopt_dim_embedding = [
    Real(        low=1e-5, high=1e-3, prior='log-uniform', name='learning_rate'     ),
    Real(        low=1e-3, high=0.1, prior='log-uniform', name='dropout'     ),
    Integer(     low=5, high=32,                          name='embedding'   ),
    Integer(     low=2, high=32,                          name='rnn_units'   ),
    Integer(     low=1,    high=5,                         name='dense_layers'      ),
    Integer(     low=10,    high=50,                         name='dense_units'      ),
    ]
embedding.set_hp(learning_rate = 0.000067, dropout = 0.003023, embedding = 16, rnn_units = 17, 
                 dense_layers = 4, dense_units = 33)
embedding.set_nlp_param(init_embedding = False)
embedding.set_skopt_dim(skopt_dim_embedding)


# 6th experiment dimred
dimred = BaseExperiment('DIMRED', batch_size = 4, max_epochs = 20, max_words = 400)
skot_dim_dimred = [
    Real(        low=1e-6, high=1e-4, prior='log-uniform', name='learning_rate'     ),
    Real(        low=1e-5, high=0.1, prior='log-uniform', name='dropout'     ),
    Integer(     low=3, high=20,                          name='embedding'   ),
    Integer(     low=10, high=32,                          name='rnn_units'   ),
    Integer(     low=10, high = 100,                       name = 'units_site'    ),
    Integer(     low=1,    high=8,                         name='dense_layers'      ),
    Integer(     low=10,    high=100,                         name='dense_units'      ),
    #Integer(     low=0,    high=1,                         name='encode_sites'      ),
    ]

dimred.set_hp(dropout = 0.001000, units_site = 100, dense_layers = 5, embedding = 5, rnn_units = 20, 
                    dense_units = 50, encode_sites = 1, learning_rate = 0.000010) #BEST
#dimred.set_hp(dropout = 0., units_site = 100, dense_layers = 3, embedding = 20, rnn_units = 10, 
#                    dense_units = 50, encode_sites = 0, learning_rate = 0.001) # nnlo
dimred.set_nlp_param(init_embedding = False)
dimred.set_skopt_dim(skot_dim_dimred)

experiments = [  nominal, dim20, avg, dim20_att, embedding, dimred  ]




