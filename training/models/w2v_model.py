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
import keras
from skopt.space import Real, Categorical, Integer



def set_avg_vec(frame):

    def avg_vec(vecs):

        avg_vecs = []
        for seq in vecs:
            if isinstance(seq, list):
                avg_vecs.append(np.array(seq))
        if len(avg_vecs) == 0:
            return np.zeros((50))
        else:
            vec = np.array(avg_vecs)[0] 
            res = np.average(vec, axis=0)
            return res
        
    frame['avg'] = frame['avg_w2v'].apply(make_avg_vec)


class SimpleAverage(BaseModel):
    
    def __init__(self, num_classes, num_error, num_sites, embedding_dim):
        
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model_params = {
            'learning_rate':0.005675,
            'dense_units':35,
            'dense_layers' : 3,
            'regulizer_value' : 0.001000,
            'dropout_value' : 0.052315
        }        
    
    
    def create_model( self, learning_rate, dense_units, dense_layers, regulizer_value, dropout_value ):
        
        m_w2v_in = Input(shape=(self.embedding_dim, ))
        m_w2v = m_w2v_in
        print( m_w2v)
        m_w2v = Dense( units=dense_units, activation='relu', 
                       kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m_w2v)
        m_w2v = Dropout(dropout_value)(m_w2v)
        print(m_w2v)
        
        m_input = Input((self.num_error,self.num_sites, 2))
        print(m_input)
        
        m = m_input

        m = Flatten()(m)
        for _ in range(dense_layers):
            m = Dense( units=dense_units, activation='relu', 
                       kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
            m = Dropout(dropout_value)(m)

        """
        m_w2v = Dense( units=dense_units, activation='relu', 
                   kernel_initializer='lecun_normal',
                   kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m_w2v)
        m_w2v = Dropout(dropout_value)(m_w2v) 
        """
        print(m)
        m_concat = Concatenate(axis=1)([m_w2v, m])
        
        m_output = Dense( units=1, activation='sigmoid', 
                          kernel_initializer='lecun_normal',
                          kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m_concat)
        
        self.model = keras.models.Model([m_input, m_w2v_in], m_output)
        self.model.compile( loss = 'binary_crossentropy', #'categorical_crossentropy',
                            optimizer = keras.optimizers.Adam(lr=learning_rate), metrics = ['accuracy'])        
        
        

class ErrorSiteAverage(BaseModel):
    
    
    def __init__(self, num_classes, num_error, num_sites, embedding_dim):
        
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model_params = {
            'dense_layers' : 6,
            'dense_units':35,
            'regulizer_value' : 0.001000,
            'dropout_value' : 0.052315,
            'learning_rate':0.005675
        }        
        
        
    def create_model( self, learning_rate, dense_units, dense_layers, regulizer_value, dropout_value ):
        
        # Input word2vec
        w2v_in = Input(shape=(self.num_error, self.num_sites, self.embedding_dim))
        # Reshape the matrix
        w2v_reshaped = Reshape(( self.num_error * self.num_sites , self.embedding_dim ))(w2v_in) 
        w2v_encoder = TimeDistributed(Dense( units=10, activation='relu',  kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) ))(w2v_reshaped)
        w2v_encoder_reshaped = Reshape(( self.num_error , self.num_sites , 10 ))(w2v_encoder) 


        # Input count
        count_in = Input((self.num_error,self.num_sites, 2))
        #count_reshaped = Reshape(( self.num_error, self.num_sites * 2 ))(count_in)

        # Concat the results
        m_concat = Concatenate(axis=3)([w2v_encoder_reshaped, count_in])
        m = m_concat
        m = Flatten()(m)
        for _ in range(dense_layers):
            m = Dense( units=dense_units, activation='relu', 
                       kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
            m = Dropout(dropout_value)(m)
        
        output = Dense( units=1, activation='sigmoid', 
                        kernel_initializer='lecun_normal',
                        kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
        
        self.model = keras.models.Model([w2v_in, count_in], output)
        self.model.compile( loss = 'binary_crossentropy', #'categorical_crossentropy',
                            optimizer = keras.optimizers.Adam(lr=learning_rate), metrics = ['accuracy'])  
    
    def get_skopt_dimensions(self):
        ''' initializes self.dimensions list
            !!! order of elements must be the same as self.create_model() params !!!
            !!! name fields must be the same as keys in self.model_params dict   !!!
        '''
        dimensions = [
            Integer(     low=1,    high=15,                        name='dense_layers'      ),
            Integer(     low=5,    high=75,                        name='dense_units'       ),
            Real(        low=1e-3, high=0.9,  prior="log-uniform", name='regulizer_value'   ),
            Real(        low=0.01, high=0.5,                       name='dropout_value'     ),
            Real(        low=1e-5, high=1e-2, prior='log-uniform', name='learning_rate'     )
        ]    
        return dimensions
    
    
    
class W2V(BaseModel):
    
    
    def __init__(self, num_classes, num_error, num_sites, embedding_dim):
        
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model_params = {
            
            'dense_layers' : 6,
            'dense_units':35,
            'regulizer_value' : 0.001000,
            'dropout_value' : 0.052315,
            'learning_rate':0.005675
        }        
        
        
    def create_model( self, dense_layers, dense_units, regulizer_value, dropout_value, learning_rate ):
        
        
        # Input word2vec
        w2v_in = Input(shape=(self.num_error, self.num_sites, self.embedding_dim))
        # Reshape the matrix
        w2v_reshaped = Reshape(( self.num_error , self.num_sites * self.embedding_dim ))(w2v_in) 
        
        w2v_encoder = TimeDistributed(Dense( units=20, activation='relu',  kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) ))(w2v_reshaped)
        #w2v_encoder_reshaped = Reshape(( self.num_error , self.num_sites , 10 ))(w2v_encoder) 

        m = Flatten()(w2v_encoder)
        for _ in range(dense_layers):
            m = Dense( units=dense_units, activation='relu', 
                       kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
            m = Dropout(dropout_value)(m)
        
        output = Dense( units=1, activation='sigmoid', 
                        kernel_initializer='lecun_normal',
                        kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
        
        self.model = keras.models.Model([w2v_in], output)
        self.model.compile( loss = 'binary_crossentropy', #'categorical_crossentropy',
                            optimizer = keras.optimizers.Adam(lr=learning_rate), metrics = ['accuracy'])   
    
    def get_skopt_dimensions(self):
        ''' initializes self.dimensions list
            !!! order of elements must be the same as self.create_model() params !!!
            !!! name fields must be the same as keys in self.model_params dict   !!!
        '''
        dimensions = [
            Integer(     low=1,    high=15,                        name='dense_layers'      ),
            Integer(     low=5,    high=75,                        name='dense_units'       ),
            Real(        low=1e-3, high=0.9,  prior="log-uniform", name='regulizer_value'   ),
            Real(        low=0.01, high=0.5,                       name='dropout_value'     ),
            Real(        low=1e-5, high=1e-2, prior='log-uniform', name='learning_rate'     )
        ]
        
        return dimensions
