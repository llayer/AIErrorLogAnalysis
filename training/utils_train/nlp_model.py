import numpy as np
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout, Flatten, Reshape
from keras.layers import average, Concatenate
from keras.models import Model

from base_model import BaseModel
from utils_train.losses import weighted_categorical_crossentropy
from utils_train.model_utils import get_class_weights


class NLP_Model(BaseModel):
    
    
    def __init__(self, num_classes, num_error, , num_sites, max_sequence_length):
        
        self.embedding_matrix = np.load('/nfshome/llayer/data/embedding_matrix.npy')
        
        self.max_sequence_length = max_sequence_length
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.max_senten_num = num_error * num_sites
        self.model_params = {
            'dense_layers':3,
            'dense_units':50,
            'regulizer_value':0.0015,
            'dropout_value':0.015,
            'learning_rate':1e-3,
        }
        
        
        
    def get_embedding_layer( self ):
        
        dims_embed = embedding_matrix.shape
        embedding = Embedding(dims_embed[0], dims_embed[1], weights=[embedding_matrix], \
                              input_length = self.max_sequence_length, trainable = False)
    
        return embedding
    
    def word_encoder_model( self ):
        
        word_input = Input(shape=(max_sequence_length,))
        word_sequences = self.get_embedding_layer()(word_input)
        word_lstm = LSTM(10)(word_sequences)
        wordEncoder = Model(word_input, word_lstm)
        
        return wordEncoder
    
    def site_encoder_model( self ):
        
        exit_code_input = Input(shape=( self.num_errors, 12, ))
        flattened = Flatten()(exit_code_input)
        dense = Dense(10, activation = "relu", kernel_initializer="normal")(flattened)
        site_encoder = Model(exit_code_input, dense)
        
        return site_encoder
    
    def create_model( self ):
        
        # Input layers
        count_input = Input(shape = (max_senten_num, 2, ))
        sent_input = Input(shape = (max_senten_num, MAX_SEQUENCE_LENGTH))
        
        # Encode the words of the sentences
        sent_encoder = TimeDistributed(self.word_encoder_model)(sent_input)
        
        # Merge the counts and words
        merge_counts_words = Concatenate(axis=2)([sent_encoder, count_input])
        
        # Reshape the tensor to wrap up the sites
        reshape_sites_codes = Reshape(( self.num_errors, self.num_sites , 12))
        reshape = reshape_sites_codes(merge_counts_words)
        
        # Encode the times 
        exit_code_encoder = TimeDistributed(self.site_encoder_model)(reshape)
        
        # Flatten
        flattened = Flatten()(exit_code_encoder)
        
        # Dense
        dense = Dense(10, activation = "relu", kernel_initializer="normal")(flattened)
        
        # Output layer
        preds = Dense(1, activation='sigmoid')(dense)
        
        # Final model
        self.model = Model([sent_input, count_input], preds)
        self.model.compile( loss = weighted_categorical_crossentropy(self.class_weights), \
                                    optimizer = keras.optimizers.Adam(lr=learning_rate) )
        
        return self.model
    
    def print_summary:
        
        print( 'Word encoder model' )
        self.word_encoder_model.summary()
        
        print( 'Site encoder model' )
        self.site_encoder_model.summary()
        
        print( 'Full model' )
        self.create_model.summary()
        
    def set_skopt_dimensions(self):
        pass
        