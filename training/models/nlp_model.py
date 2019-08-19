import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout, Flatten, Reshape
from keras.layers import average, Concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adam
from models.base_model import BaseModel
from keras import backend as K

class NLP_Model(BaseModel):
    
    
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
            'learning_rate':0.1
        }
        
        
        
    def get_embedding_layer( self ):
        
        dims_embed = self.embedding_matrix.shape
        embedding = Embedding(dims_embed[0], dims_embed[1], weights=[self.embedding_matrix], \
                              input_length = self.max_sequence_length, trainable = False)
    
        return embedding
    
    def word_encoder_model( self ):
        
        word_input = Input(shape = (self.max_sequence_length, ), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)
        word_lstm = LSTM(10)(word_sequences)
        wordEncoder = Model(word_input, word_lstm)
        
        return wordEncoder
    
    def sentence_encoder_model( self ):
        pass
        
    
    def site_encoder_model( self ):
        
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
        sent_encoder = TimeDistributed(self.word_encoder_model())(sent_input_reshaped)
        
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
        exit_code_encoder = TimeDistributed(self.site_encoder_model())(merge_counts_words)
        
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
        word_encoder = self.word_encoder_model()
        word_encoder.summary()
        print()
        print()
        
        print( 'Site encoder model' )
        site_encoder = self.site_encoder_model()
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
    
    
    
    
    
class NLP_SingleMsg(BaseModel):
    
    
    def __init__(self, num_classes, num_error, num_sites, max_sequence_length, debug = False):
        
        self.embedding_matrix = np.load('/nfshome/llayer/data/embedding_matrix.npy')
        
        self.max_sequence_length = max_sequence_length
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.max_senten_num = num_error * num_sites
        self.debug = debug
        self.model_params = {
            'learning_rate':0.1
        }
        
        
        
    def get_embedding_layer( self ):
        
        dims_embed = self.embedding_matrix.shape
        embedding = Embedding(dims_embed[0], dims_embed[1], weights=[self.embedding_matrix], \
                              input_length = self.max_sequence_length, trainable = False)
    
        return embedding
    
    def word_encoder_model( self ):
        
        word_input = Input(shape = (self.max_sequence_length, ), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)
        word_lstm = LSTM(10)(word_sequences)
        wordEncoder = Model(word_input, word_lstm)
        
        return wordEncoder
    
    def sentence_encoder_model( self ):
        pass
        
    
    def site_encoder_model( self ):
        
        exit_code_input = Input(shape=( self.num_sites, 12, ), dtype='float32')
        flattened = Flatten()(exit_code_input)
        dense = Dense(10, activation = "relu", kernel_initializer="normal")(flattened)
        site_encoder = Model(exit_code_input, dense)
        
        return site_encoder
    
    def create_model( self, learning_rate ):
        
        # Input layers
        sent_input = Input(shape = (self.num_error, self.num_sites, self.max_sequence_length), dtype='float32')
        
        if self.debug: print( sent_input )
        
        # Reshape the matrix
        sent_input_reshaped = Reshape(( self.num_error * self.num_sites, self.max_sequence_length))(sent_input)
        
        if self.debug: print( sent_input_reshaped )
        
        # Encode the words of the sentences
        sent_encoder = TimeDistributed(self.word_encoder_model())(sent_input_reshaped)
        
        if self.debug: print( sent_encoder )
        
        # Shape back to concat the matrix
        sent_encoder_reshaped = Reshape(( self.num_error * self.num_sites, 10))(sent_encoder)
        
        if self.debug: print( sent_encoder_reshaped )
  
            
        # Reshape the error sites matrix
        sentence_averaged_reshaped = Reshape(( self.num_error , self.num_sites, 10))(sentence_averaged)
        
        if self.debug: print( sentence_averaged_reshaped )
        
        # Reshape the tensor to wrap up the sites
        #reshape_sites_codes = Reshape(( self.num_error, self.num_sites , 12))
        #reshape = reshape_sites_codes(merge_counts_words)
        
        # Encode the sites 
        exit_code_encoder = TimeDistributed(self.site_encoder_model())(sentence_averaged_reshaped)
        
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
        self.model = Model(sent_input, preds)
        self.model.compile( loss='binary_crossentropy',  #weighted_categorical_crossentropy(self.class_weights), \
                                    optimizer = Adam(lr = learning_rate) )
        
        return self.model
    
    def print_summary(self):
        
        print( 'Word encoder model' )
        print()
        word_encoder = self.word_encoder_model()
        word_encoder.summary()
        print()
        print()
        
        print( 'Site encoder model' )
        site_encoder = self.site_encoder_model()
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
            
    
    
    
    
    
        