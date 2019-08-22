import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import keras
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout, Flatten, Reshape
from keras.layers import average, Concatenate, Lambda, CuDNNLSTM, CuDNNGRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D
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
    
    
    def __init__(self, num_classes, num_error, num_sites, max_sequence_length, cudnn = False, batch_norm = False, 
                 train_embedding = False, word_encoder = 'LSTM', encode_sites = True, attention = False,
                 include_counts = False, verbose = 1):
        
        embedding_matrix = np.load('/nfshome/llayer/data/embedding_matrix.npy')
        self.embedding_matrix = embedding_matrix.astype('float32')
        self.max_sequence_length = max_sequence_length
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.max_senten_num = num_error * num_sites
        self.cudnn = cudnn
        self.attention = attention
        self.word_encoder = word_encoder
        self.encode_sites = encode_sites
        self.batch_norm = batch_norm
        self.include_counts = include_counts
        self.train_embedding = train_embedding
        self.verbose = verbose
        # Hyperparameters
        self.hp = {
            # Regularization
            'l2_regulizer': keras.regularizers.l2(0.0001),
            'dropout':0.2,
            # Conv1D
            'filters':256,
            'kernel_size':3,
            'conv_layers':3,
            'max_pooling':3,
            'units_conv':10,
            # RNN with optional attention
            'att_units':10,
            'rec_dropout':0.0,
            'rnn': LSTM, #TRY GRU
            'rnncud': CuDNNLSTM, # TRY CuDNNGRU
            'rnn_units' : 10,
            'rnn_dropout': None,
            # Site encoding
            'activation_site': 'relu', #TRY linear
            'units_site': 10,
            # Final layers
            'dense_layers': 3,
            'dense_units': 20,
            'learning_rate':0.0001
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
        if self.cudnn == True or self.word_encoder == 'Conv1D':
            embedding = Embedding(dims_embed[0], dims_embed[1], weights=[self.embedding_matrix], \
                                  input_length = self.max_sequence_length, trainable = self.train_embedding)
        else:
            embedding = Embedding(dims_embed[0], dims_embed[1], weights=[self.embedding_matrix], \
                      input_length = self.max_sequence_length, mask_zero = True, trainable = self.train_embedding)
    
        return embedding
    
    
    def word_encoder_lstm( self ):
        
        #TODO add recurrent_dropout
        
        word_input = Input(shape = ( self.max_sequence_length, ), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)
                
        if self.attention == False:
            if self.cudnn == True:
                word_lstm = self.hp['rnncud'](self.hp['rnn_units'], kernel_regularizer=self.hp['l2_regulizer'])(word_sequences)
            else:
                word_lstm = self.hp['rnn'](self.hp['rnn_units'], kernel_regularizer=self.hp['l2_regulizer'],
                                          recurrent_dropout = self.hp['rec_dropout'])(word_sequences)
            wordEncoder = Model(word_input, word_lstm)
        else:
            if self.cudnn == True:
                word_lstm = self.hp['rnncud'](self.hp['rnn_units'], kernel_regularizer=self.hp['l2_regulizer'],
                                             return_sequences=True)(word_sequences)
            else:
                word_lstm = self.hp['rnn'](self.hp['rnn_units'], kernel_regularizer=self.hp['l2_regulizer'],
                                          recurrent_dropout = self.hp['rec_dropout'], return_sequences=True)(word_sequences)
            word_dense = TimeDistributed(Dense(self.hp['att_units']), kernel_regularizer=self.hp['l2_regulizer'])(word_lstm)
            word_att = AttentionWithContext()(word_dense)
            wordEncoder = Model(word_input, word_att)
        
        return wordEncoder
    

    def word_encoder_conv( self ):
        
        #TODO add spatial dropout
        
        word_input = Input(shape = ( self.max_sequence_length, ), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)

        for i in range(self.hp['conv_layers']):
            word_sequences = Conv1D(self.hp['filters'], self.hp['kernel_size'], 
                                    activation='relu',kernel_regularizer=self.hp['l2_regulizer'])(word_sequences)
            word_sequences = MaxPooling1D(self.hp['max_pooling'])(word_sequences)

        word_sequences = GlobalMaxPooling1D()(word_sequences)
        word_sequences = Dense(self.hp['units_conv'], activation='relu',
                               kernel_regularizer=self.hp['l2_regulizer'])(word_sequences)
        
        wordEncoder = Model(word_input, word_sequences)

        return wordEncoder
    
    """
    def site_word_encoder( self ): #, units ):
        
        
        exit_code_input = Input(shape=( self.num_sites, units ), dtype='float32')
        print( units )
        print( exit_code_input )
        flat = Flatten()(exit_code_input)
        print( flat )
        exit_code = Dense(self.hp['units_site'], activation = self.hp['activation_site'], 
                      kernel_regularizer=self.hp['l2_regulizer'])(flat)
        print( exit_code )
        site_encoder = Model(exit_code_input, exit_code)
        
        return site_encoder
        
        exit_code_input = Input(shape=( self.num_sites, 12, ), dtype='float32')
        flattened = Flatten()(exit_code_input)
        d = Dense(10, activation = "relu", kernel_initializer="normal")(flattened)
        site_encoder = Model(exit_code_input, d)
        return site_encoder
    """
        
    def create_model( self ):
        
        if self.verbose == 1:
            self.print_hyperparameters()
        
        # Input layers
        sent_input = Input(shape = (self.num_error, self.num_sites, self.max_sequence_length), dtype='float32')
        
        # Reshape the matrix
        sent_input_reshaped = Reshape(( self.num_error * self.num_sites, self.max_sequence_length))(sent_input)
        
        # Encode the words of the sentences
        if self.word_encoder == 'LSTM':
            encoder_units = self.hp['rnn_units']
            sent_encoder = TimeDistributed(self.word_encoder_lstm())(sent_input_reshaped)
        elif self.word_encoder == 'Conv1D':
            encoder_units = self.hp['units_conv']
            sent_encoder = TimeDistributed(self.word_encoder_conv())(sent_input_reshaped)
        else: 
            print( 'No valid encoder' )    
            
        sent_encoder = Dropout(self.hp['dropout'])(sent_encoder)
        if self.batch_norm == True:
            sent_encoder = BatchNormalization()(sent_encoder)
            
        # Reshape the error sites matrix
    
        sent_encoder_reshaped = Reshape(( self.num_error , self.num_sites, encoder_units))(sent_encoder)
        
        # Add the meta information
        if self.include_counts == True:
            
            count_input = Input(shape = (self.num_error, self.num_sites, 2, ), dtype='float32')
            # Merge the counts and words
            exit_code_site_repr = Concatenate(axis=3)([sent_encoder_reshaped, count_input])
        
        else:
            exit_code_site_repr = sent_encoder_reshaped
        
        # Encode the site
        if self.encode_sites == True:
            
            exit_code_site_repr = Reshape(( self.num_error , self.num_sites * encoder_units))(exit_code_site_repr)
            exit_code_encoder = TimeDistributed(Dense(self.hp['units_site'], activation = self.hp['activation_site'], 
                      kernel_regularizer=self.hp['l2_regulizer']))(exit_code_site_repr)

            exit_code_encoder = Dropout(self.hp['dropout'])(exit_code_encoder)
            if self.batch_norm == True:
                exit_code_encoder = BatchNormalization()(exit_code_encoder)
            
        # Flatten
        flattened = Flatten()(exit_code_encoder)
            
        # Dense
        dense = flattened
        for _ in range(self.hp['dense_layers']):
            
            dense = Dense( units=self.hp['dense_units'], activation='relu', kernel_regularizer=self.hp['l2_regulizer'] )(dense)
            dense = Dropout(self.hp['dropout'])(dense)
            if self.batch_norm == True:
                dense = BatchNormalization()(dense)            
            
        # Output layer
        preds = Dense(1, activation='sigmoid', kernel_regularizer=self.hp['l2_regulizer'] )(dense)
                
        # Final model
        self.model = Model(sent_input, preds)
        self.model.compile( loss='binary_crossentropy', optimizer = Adam(lr = self.hp['learning_rate']) )
        
        return self.model
 
    
    def print_summary(self):
        
        print( 'Word encoder model' )
        print()
        word_encoder = self.word_word_encoder()
        word_encoder.summary()
        print()
        print()
        
        print( 'Site encoder model' )
        site_encoder = self.site_word_encoder()
        site_encoder.summary()
        print()
        print()
        
        print( 'Full model' )
        model = self.model
        model.summary()
        print()
        print()
        
    def get_skopt_dimensions(self):
        
        dimensions = [
            Real(        low=1e-5, high=1e-3, prior='log-uniform', name='learning_rate'     )
        ]
        return dimensions

"""
class NLP_Sequence(BaseModel):
    
    
    def __init__(self, num_classes, num_error, num_sites, max_sequence_length, max_msg, debug = False):
        
        self.embedding_matrix = np.load('/nfshome/llayer/data/embedding_matrix.npy')
        
        self.max_sequence_length = max_sequence_length
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.max_senten_num = num_error * num_sites
        self.max_msg = max_msg
        self.debug = debug
        self.model_params = {
            'learning_rate':0.001
        }
        
        
        
    def get_embedding_layer( self ):
        
        dims_embed = self.embedding_matrix.shape
        embedding = Embedding(dims_embed[0], dims_embed[1], weights=[self.embedding_matrix], \
                              input_length = self.max_sequence_length, trainable = False)
    
        return embedding
    
    def word_word_encoder( self ):
        
        word_input = Input(shape = (self.max_sequence_length, ), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)
        word_lstm = CuDNNLSTM(10)(word_sequences)
        #print word_lstm
        wordEncoder = Model(word_input, word_lstm)
        
        return wordEncoder
    
    def sentence_word_encoder( self ):
        pass
        
    
    def site_word_encoder( self ):
        
        exit_code_input = Input(shape=( self.num_sites, 12, ), dtype='float32')
        flattened = Flatten()(exit_code_input)
        dense = Dense(10, activation = "relu", kernel_initializer="normal")(flattened)
        site_encoder = Model(exit_code_input, dense)
        
        return site_encoder
    
    def create_model( self, learning_rate ):
        
        # Input layers
        count_input = Input(shape = (self.num_error, self.num_sites, 2, ), dtype='float32')
        sent_input = Input(shape = (self.num_error, self.num_sites, self.max_msg, self.max_sequence_length), dtype='float32')
        
        if self.debug: print( sent_input )
        
        # Reshape the matrix
        sent_input_reshaped = Reshape(( self.num_error * self.num_sites * self.max_msg , self.max_sequence_length))(sent_input)
        
        if self.debug: print( sent_input_reshaped )
        
        # Encode the words of the sentences
        sent_encoder = TimeDistributed(self.word_word_encoder())(sent_input_reshaped)
        
        if self.debug: print( sent_encoder )
        
        # Shape back to concat the matrix
        sent_encoder_reshaped = Reshape(( self.num_error * self.num_sites, self.max_msg , 10))(sent_encoder)
        
        if self.debug: print( sent_encoder_reshaped )
        
        # Average the message sequence
        sentence_averaged = Lambda(lambda x: K.mean(x, axis=2))(sent_encoder_reshaped)
        
        if self.debug: print( sentence_averaged )
            
        # Reshape the error sites matrix
        
        sentence_averaged_reshaped = Reshape(( self.num_error , self.num_sites, 10))(sentence_averaged)
        
        if self.debug: print( sentence_averaged_reshaped )
        if self.debug: print( count_input )
        
        # Merge the counts and words
        merge_counts_words = Concatenate(axis=3)([sentence_averaged_reshaped, count_input])
        
        if self.debug: print( merge_counts_words )
        
        # Reshape the tensor to wrap up the sites
        #reshape_sites_codes = Reshape(( self.num_error, self.num_sites , 12))
        #reshape = reshape_sites_codes(merge_counts_words)
        
        # Encode the sites 
        exit_code_encoder = TimeDistributed(self.site_word_encoder())(merge_counts_words)
        
        if self.debug: print( exit_code_encoder )
        
        # Flatten
        flattened = Flatten()(exit_code_encoder)
        
        if self.debug: print( flattened )
        
        # Dense
        dense = Dense(10, activation = "relu", kernel_initializer="normal")(flattened)
        
        if self.debug: print( dense )
        
        # Output layer
        preds = Dense(1, activation='sigmoid')(dense)
        
        if self.debug: print(preds)
        
        # Final model
        self.model = Model([sent_input, count_input], preds)
        self.model.compile( loss='binary_crossentropy',  #weighted_categorical_crossentropy(self.class_weights), \
                                    optimizer = Adam(lr = learning_rate) )
        
        return self.model
    
    def print_summary(self):
        
        print( 'Word encoder model' )
        print()
        word_encoder = self.word_word_encoder()
        word_encoder.summary()
        print()
        print()
        
        print( 'Site encoder model' )
        site_encoder = self.site_word_encoder()
        site_encoder.summary()
        print()
        print()
        
        print( 'Full model' )
        model = self.model
        model.summary()
        print()
        print()
        
    def set_skopt_dimensions(self):
        pass
    
"""
    
    

    
    
    
    
    
        
