import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import keras
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout, Flatten, Reshape
from keras.layers import average, Concatenate, Lambda, CuDNNLSTM, CuDNNGRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D, AveragePooling1D
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from models.base_model import BaseModel
from models.attention_context import AttentionWithContext
from keras import backend as K
from keras.layers import BatchNormalization
from skopt.space import Real, Categorical, Integer
K.set_floatx('float32')
print(K.floatx())


class NLP(BaseModel):
    
    
    def __init__(self, num_classes, num_error, num_sites, max_sequence_length, embedding_matrix_path,
                 cudnn = False, batch_norm = False, word_encoder = 'LSTM', encode_sites = True, attention = False,
                 include_counts = False, avg_w2v = False, init_embedding = True, verbose = 1):
        
        embedding_matrix = np.load(embedding_matrix_path)
        self.embedding_matrix = embedding_matrix.astype('float32')
        self.max_sequence_length = max_sequence_length
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.max_senten_num = num_error * num_sites
        self.cudnn = cudnn
        self.attention = attention
        self.init_embedding = init_embedding
        self.word_encoder = word_encoder
        self.encode_sites = encode_sites
        self.batch_norm = batch_norm
        self.include_counts = include_counts
        self.avg_w2v = avg_w2v
        self.verbose = verbose
        # Hyperparameters
        self.hp = {
            # Regularization
            'l2_regulizer': 0.0001,
            'dropout':0.2,
            # Conv1D
            'filters':256,
            'kernel_size':3,
            'conv_layers':3,
            'max_pooling':3,
            'units_conv':10,
            # RNN with optional attention
            'train_embedding': False,
            'att_units':15,
            'rec_dropout':0.0,
            'rnn': LSTM, #TRY GRU
            'rnncud': CuDNNLSTM, # TRY CuDNNGRU
            'rnn_units' : 10,
            'embedding': 20,
            # Site encoding
            'encode_sites': False,
            'activation_site': 'relu', #TRY linear
            'units_site': 10,
            'pool_size':2,
            # Final layers
            'dense_layers': 3,
            'dense_units': 20,
            'learning_rate':0.0001,
            'decay':0.0
                    }

        
        
    def set_hyperparameters(self, tweaked_instances):

        for  key, value in tweaked_instances.items():
            if key in self.hp:
                self.hp[key] = value
            else:
                raise KeyError(key + ' does not exist in hyperparameters')

            
    def print_hyperparameters(self):

        print('Hyperparameter\tCorresponding Value')
        for key, value in self.hp.items():
            print(key, '\t\t', value)
        
        
    def get_embedding_layer( self ):
        
        dims_embed = self.embedding_matrix.shape
        print ( 'Embedding', self.init_embedding )
        if self.init_embedding == True:
            
            """
            if self.cudnn == True or self.word_encoder == 'Conv1D':
                embedding = Embedding(dims_embed[0], dims_embed[1], weights=[self.embedding_matrix], \
                                      input_length = self.max_sequence_length, trainable = self.train_embedding)
            else:
            """

            embedding = Embedding(dims_embed[0], dims_embed[1], weights=[self.embedding_matrix], \
                      input_length = None, mask_zero = True, trainable = int(self.hp['train_embedding']))
        else:
            print( 'Embedding', self.hp['embedding'] )
            embedding = Embedding(dims_embed[0], int(self.hp['embedding']), mask_zero = True)            
            
        print( embedding )
        return embedding
    
    
    def word_encoder_lstm( self ):
        
        #TODO add recurrent_dropout
        
        word_input = Input(shape = ( None, ), dtype='int32')
        word_sequences = self.get_embedding_layer()(word_input)
                
        if self.attention == False:
            if self.cudnn == True:
                word_lstm = self.hp['rnncud'](int(self.hp['rnn_units']), 
                                              kernel_regularizer=l2(self.hp['l2_regulizer']))(word_sequences)
            else:
                word_lstm = self.hp['rnn'](int(self.hp['rnn_units']), kernel_regularizer=l2(self.hp['l2_regulizer']),
                                          recurrent_dropout = self.hp['rec_dropout'])(word_sequences)
            #word_lstm = Dense(5, activation = "relu", kernel_initializer="normal")(word_lstm)
            wordEncoder = Model(word_input, word_lstm)
        else:
            if self.cudnn == True:
                word_lstm = self.hp['rnncud'](int(self.hp['rnn_units']), kernel_regularizer=l2(self.hp['l2_regulizer']),
                                             return_sequences=True)(word_sequences)
            else:
                word_lstm = self.hp['rnn'](int(self.hp['rnn_units']), kernel_regularizer=l2(self.hp['l2_regulizer']),
                                          recurrent_dropout = self.hp['rec_dropout'], return_sequences=True)(word_sequences)
            word_dense = TimeDistributed(Dense(int(self.hp['att_units'])))(word_lstm)
            word_att = AttentionWithContext()(word_dense)
            wordEncoder = Model(word_input, word_att)
        if self.verbose == 1:
            wordEncoder.summary()     
        return wordEncoder
    

    def word_encoder_conv( self ):
        
        #TODO add spatial dropout
        
        word_input = Input(shape = ( self.max_sequence_length, ), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)

        for i in range(self.hp['conv_layers']):
            word_sequences = Conv1D(self.hp['filters'], self.hp['kernel_size'], 
                                    activation='relu',kernel_regularizer=l2(self.hp['l2_regulizer']))(word_sequences)
            word_sequences = MaxPooling1D(self.hp['max_pooling'])(word_sequences)

        word_sequences = GlobalMaxPooling1D()(word_sequences)
        word_sequences = Dense(self.hp['units_conv'], activation='relu',
                               kernel_regularizer=l2(self.hp['l2_regulizer']))(word_sequences)
        
        wordEncoder = Model(word_input, word_sequences)

        return wordEncoder

        
    def create_model( self ):
        
        if self.verbose == 1:
            self.print_hyperparameters()
        
        
        # Input layers
        #sent_input = Input(shape = (self.num_error, self.num_sites, None), dtype='int32')
        
        # Reshape the matrix
        #sent_input_reshaped = Reshape(( self.num_error * self.num_sites, ))(sent_input)
       
        if self.avg_w2v == False:
            
            sent_input = Input(shape = (self.num_error * self.num_sites, None), dtype='int32')
            
            # Encode the words of the sentences
            if self.word_encoder == 'LSTM':
                if self.attention == False:
                    encoder_units = int(self.hp['rnn_units'])
                else:
                    encoder_units = int(self.hp['att_units'])
                sent_encoder = TimeDistributed(self.word_encoder_lstm())(sent_input)
            elif self.word_encoder == 'Conv1D':
                encoder_units = self.hp['units_conv']
                sent_encoder = TimeDistributed(self.word_encoder_conv())(sent_input_reshaped)
            else: 
                print( 'No valid encoder' )    


            """    
            sent_encoder = Dropout(self.hp['dropout'])(sent_encoder)
            if self.batch_norm == True:
                sent_encoder = BatchNormalization()(sent_encoder)
            """

            # Reshape the error sites matrix

            sent_encoder_reshaped = Reshape(( self.num_error , self.num_sites, encoder_units))(sent_encoder)
         
        else:
            
            #sent_input = Input(shape = (self.num_error * self.num_sites, self.max_sequence_length), dtype='float32')
            #encoder_units = int(self.max_sequence_length / int(self.hp['pool_size']))
            #sent_pool = AveragePooling1D(pool_size = int(self.hp['pool_size']), data_format='channels_first')(sent_input)
            #sent_encoder_reshaped = Reshape(( self.num_error , self.num_sites, encoder_units))(sent_pool)
            #sent_encoder_reshaped = Reshape(( self.num_error , self.num_sites, self.max_sequence_length))(sent_pool)
            sent_input = Input(shape = (self.num_error * self.num_sites, self.max_sequence_length), dtype='float32')
            sent_encoder_reshaped = Reshape(( self.num_error , self.num_sites, self.max_sequence_length))(sent_input)
            sent_encoder_reshaped = TimeDistributed(Dense(int(self.hp['units_site']), activation = self.hp['activation_site'], 
                       kernel_regularizer=l2(self.hp['l2_regulizer'])))(sent_encoder_reshaped)
            encoder_units = int(self.hp['units_site'])
            #encoder_units = self.max_sequence_length
        
        # Add the meta information
        if self.include_counts == True:
            
            count_input = Input(shape = (self.num_error, self.num_sites, 2, ), dtype='float32')
            print( count_input )
            # Merge the counts and words
            exit_code_site_repr = Concatenate(axis=3)([sent_encoder_reshaped, count_input])
            print( exit_code_site_repr )
            exit_code_site_repr = Reshape(( self.num_error , self.num_sites * (encoder_units+2)))(exit_code_site_repr)
            print( exit_code_site_repr )
        else:
            exit_code_site_repr = sent_encoder_reshaped
            exit_code_site_repr = Reshape(( self.num_error , self.num_sites * (encoder_units)))(exit_code_site_repr)
        
        
        # Encode the site
        if int(self.hp['encode_sites']) == True:
            
            exit_code_encoder = TimeDistributed(Dense(int(self.hp['units_site']), activation = self.hp['activation_site'], 
                      kernel_regularizer=l2(self.hp['l2_regulizer'])))(exit_code_site_repr)
        else:
            exit_code_encoder = exit_code_site_repr

            """
            exit_code_encoder = Dropout(self.hp['dropout'])(exit_code_encoder)
            if self.batch_norm == True:
                exit_code_encoder = BatchNormalization()(exit_code_encoder)
            """

        #exit_code_encoder = AveragePooling1D(pool_size = 2, data_format='channels_first')(exit_code_site_repr)
            
        # Flatten
        flattened = Flatten()(exit_code_encoder)
            
        # Dense
        dense = flattened
        for _ in range(int(self.hp['dense_layers'])):
            
            dense = Dense( units=int(self.hp['dense_units']), activation='relu', 
                          kernel_regularizer=l2(self.hp['l2_regulizer']) )(dense)
            dense = Dropout(self.hp['dropout'])(dense)
            if self.batch_norm == True:
                dense = BatchNormalization()(dense)            
            
        # Output layer
        preds = Dense(1, activation='sigmoid', kernel_regularizer=l2(self.hp['l2_regulizer']) )(dense)
                
        # Final model
        if self.include_counts == False:
            self.model = Model(sent_input, preds)
        else:
            self.model = Model([sent_input, count_input], preds)
        self.model.compile( loss='binary_crossentropy', optimizer = Adam(lr = self.hp['learning_rate'], 
                                                                         decay = self.hp['decay']) )
        
        if self.verbose == 1:
            self.model.summary()
        
        return self.model
 
