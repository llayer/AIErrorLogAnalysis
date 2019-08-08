import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import setGPU
import keras
from keras.layers import Input, Flatten, Dense, Dropout, Reshape, multiply
import numpy as np
from skopt.space import Real, Categorical, Integer
from models.base_model import BaseModel

class FF(BaseModel):
    
    def __init__(self, num_classes, num_error=54, num_sites=142):

        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        #self.dense_units = 20
        #self.regulizer_value = 0.0015
        #self.dropout_value = 0.015
        """ 
        self.model_params = {
            'learning_rate':1e-3,
            'dense_units':20,
            'dense_layers' : 3,
            'regulizer_value' : 0.0015,
            'dropout_value' : 0.015
            
        }"""
        self.model_params = {
            'learning_rate':1e-2,
            'dense_units':20,
            'dense_layers' : 3,
            'regulizer_value' : 0.0015,
            'dropout_value' : 0.015
        }

    def create_model( self, learning_rate, dense_units, dense_layers, regulizer_value, dropout_value ): 

        m_input = Input((self.num_error,self.num_sites, 2))
        #m_input = Input((self.num_error,self.num_sites))
    
        m = m_input

        m = Flatten()(m)
        for _ in range(dense_layers):
            m = Dense( units=dense_units, activation='relu', 
                       kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
            m = Dropout(dropout_value)(m)

        #m_output = Dense( units=self.num_classes, activation='softmax', 
        #                  kernel_initializer='lecun_normal',
        #                  kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)

        m_output = Dense( units=1, activation='sigmoid', 
                          kernel_initializer='lecun_normal',
                          kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
        
        self.model = keras.models.Model(inputs=m_input, outputs=m_output)
        self.model.compile( loss = 'binary_crossentropy', #'categorical_crossentropy',
                            optimizer = keras.optimizers.Adam(lr=learning_rate), metrics = ['accuracy'])
        


    def get_skopt_dimensions(self):

        dimensions = [
            Real(        low=1e-4, high=1e-1, prior='log-uniform', name='learning_rate'     ),
            Integer(     low=10,    high=100,                        name='dense_units'       ),
            Integer(     low=2,    high=8,                        name='dense_layers'      ),
            Real(        low=1e-3, high=0.9,  prior="log-uniform", name='regulizer_value'   ),
            Real(        low=0.01, high=0.5,                       name='dropout_value'     ),
            Integer(     low=500,   high = 5000,                    name='batch_size'       )
        ]
        
        """
        self.dimensions = [
            Integer(     low=1,    high=5,                        name='dense_layers'      ),
            Integer(     low=5,    high=75,                        name='dense_units'       ),
            Real(        low=1e-3, high=0.9,  prior="log-uniform", name='regulizer_value'   ),
            Real(        low=0.01, high=0.5,                       name='dropout_value'     ),
            Real(        low=1e-4, high=1e-1, prior='log-uniform', name='learning_rate'     )
        ]
        """
        
        return dimensions

