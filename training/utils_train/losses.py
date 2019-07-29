'''
    Contains functions that invoked with or without parameters returns
    a function with inputs (y_true, y_pred), 
    y_true, y_pred shapes are (num_examples, num_classes)
'''

import keras.backend as K


def weighted_categorical_crossentropy(weights):
    ''' returns loss function with applied weights '''
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss


def categorical_crossentropy():
    def loss(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred)
    return loss


def binary_crossentropy():
    def loss(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss


def weighted_binary_crossentropy(weights):
    one_weight = K.variable(weights[0])
    zero_weight = K.variable(weights[1])
    def loss(y_true, y_pred):
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        return K.mean(weight_vector * K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss