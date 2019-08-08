import numpy as np
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout, Flatten, Reshape
from keras.layers import average, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from base_model import BaseModel
from utils_train.losses import weighted_categorical_crossentropy
from utils_train.model_utils import get_class_weights


class NLP_Model(BaseModel):
    
    
    def __init__(self, num_classes, num_error, num_sites, max_sequence_length):
        
        self.embedding_matrix = np.load('/nfshome/llayer/data/embedding_matrix.npy')
        
        self.max_sequence_length = max_sequence_length
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.max_senten_num = num_error * num_sites
        self.model_params = {
            'learning_rate':0.1
        }
        
        
        
    def get_embedding_layer( self ):
        
        dims_embed = self.embedding_matrix.shape
        embedding = Embedding(dims_embed[0], dims_embed[1], weights=[self.embedding_matrix], \
                              input_length = self.max_sequence_length, trainable = True)
    
        return embedding
    
    def word_encoder_model( self ):
        
        word_input = Input(shape = (self.max_sequence_length, ), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)
        word_lstm = LSTM(10)(word_sequences)
        wordEncoder = Model(word_input, word_lstm)
        
        return wordEncoder
    
    def site_encoder_model( self ):
        
        exit_code_input = Input(shape=( self.num_sites, 12, ), dtype='float32')
        flattened = Flatten()(exit_code_input)
        dense = Dense(10, activation = "relu", kernel_initializer="normal")(flattened)
        site_encoder = Model(exit_code_input, dense)
        
        return site_encoder
    
    def create_model( self, learning_rate ):
        
        # Input layers
        count_input = Input(shape = (self.num_error, self.num_sites, 2, ), dtype='float32')
        sent_input = Input(shape = (self.num_error, self.num_sites, self.max_sequence_length), dtype='float32')
        
        # Reshape the matrix
        sent_input_reshaped = Reshape(( self.num_error * self.num_sites , self.max_sequence_length))(sent_input)
        
        # Encode the words of the sentences
        sent_encoder = TimeDistributed(self.word_encoder_model())(sent_input_reshaped)
        
        # Shape back to concat the matrix
        sent_encoder_reshaped = Reshape(( self.num_error, self.num_sites , 10))(sent_encoder)
        
        # Merge the counts and words
        merge_counts_words = Concatenate(axis=3)([sent_encoder_reshaped, count_input])
        
        # Reshape the tensor to wrap up the sites
        #reshape_sites_codes = Reshape(( self.num_error, self.num_sites , 12))
        #reshape = reshape_sites_codes(merge_counts_words)
        
        # Encode the sites 
        exit_code_encoder = TimeDistributed(self.site_encoder_model())(merge_counts_words)
        
        # Flatten
        flattened = Flatten()(exit_code_encoder)
        
        # Dense
        dense = Dense(10, activation = "relu", kernel_initializer="normal")(flattened)
        
        # Output layer
        preds = Dense(1, activation='sigmoid')(dense)
        
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
        