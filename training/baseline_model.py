import keras
from keras.layers import Input, Flatten, Dense, Dropout, Reshape, multiply
import numpy as np
from skopt.space import Real, Categorical, Integer

from base_model import BaseModel
from utils_train.losses import weighted_categorical_crossentropy
from utils_train.model_utils import get_class_weights


class FF(BaseModel):
    
    def __init__(self, num_classes, num_error=54, num_sites=142):

        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.model_params = {
            'dense_layers':3,
            'dense_units':50,
            'regulizer_value':0.0015,
            'dropout_value':0.015,
            'learning_rate':1e-3,
        }


    def create_model( self, dense_layers, dense_units, regulizer_value, 
                      dropout_value, learning_rate ):

        m_input = Input((self.num_error,self.num_sites, 2))
        
        m = m_input

        m = Flatten()(m)
        for _ in range(dense_layers):
            m = Dense( units=dense_units, activation='relu', 
                       kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
            m = Dropout(dropout_value)(m)

        m_output = Dense( units=self.num_classes, activation='softmax', 
                          kernel_initializer='lecun_normal',
                          kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
        
        self.model = keras.models.Model(inputs=m_input, outputs=m_output)
        self.model.compile( loss = weighted_categorical_crossentropy(self.class_weights),
                            optimizer = keras.optimizers.Adam(lr=learning_rate) )
        


    def set_skopt_dimensions(self):

        self.dimensions = [
            Integer(     low=1,    high=15,                        name='dense_layers'      ),
            Integer(     low=5,    high=75,                        name='dense_units'       ),
            Real(        low=1e-3, high=0.9,  prior="log-uniform", name='regulizer_value'   ),
            Real(        low=0.01, high=0.5,                       name='dropout_value'     ),
            Real(        low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'     )
        ]


