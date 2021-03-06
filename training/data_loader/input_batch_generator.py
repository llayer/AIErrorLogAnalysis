import pandas as pd
import itertools
import math
import numpy as np
from keras.utils import to_categorical



class InputBatchGenerator(object):
    
    def __init__(self, frame, label, codes, sites, max_words, batch_size = 1, max_msg = 5, 
                 averaged = False, sequence = False, only_msg = False, cut_front = True):
        
        self.frame = frame
        self.n_tasks = len(frame)
        self.label = label
        self.batch_size = batch_size
        self.codes = codes
        self.sites = sites
        self.max_words = max_words
        if sequence == False:
            self.max_msg = 1
        else:
            self.max_msg = max_msg
        self.averaged = averaged
        self.sequence = sequence
        self.cut_front = cut_front
        self.only_msg = only_msg
        self.unique_sites = len(list(set(self.sites.values())))
        self.unique_codes = len(list(set(self.codes.values())))
        self.n_tasks = len(frame)
       
    
    

    def pad_along_axis(self, array, axis=0):

        array = np.array(array)
        pad_size = self.pad_dim - array.shape[axis]
        axis_nb = len(array.shape)

        if pad_size < 0:
            if self.cut_front == True:
                return array[-self.pad_dim : ]
            else:
                return array[ : self.pad_dim ]

        npad = [(0, 0) for x in range(axis_nb)]
        npad[axis] = (0, pad_size)

        b = np.pad(array, pad_width=npad, mode='constant', constant_values=int(0))

        return b
    
    
    def fill_counts(self, index, error, site, site_state, count):
        
        # Encode good and bad sites
        if site_state == 'good':
            site_state_encoded = 0
        else:
            site_state_encoded = 1

        self.error_site_counts[index, self.codes[error], self.sites[site], site_state_encoded] += count
    
    
    def fill_first_message(self, index, error, site, error_message):
        
                               
        # Pad the error message
        if self.averaged == False:
            error_message = self.pad_along_axis(error_message)
        #print( error_message )
        self.error_site_tokens[index, self.codes[error], self.sites[site]] = error_message

    
 
    def fill_messages_sequence(self, index, error, site, error_message_sequence):
        
        # Loop over the error message sequence
        for counter, error_message in enumerate(error_message_sequence):
           
            # Stop when maximal message is reached
            if counter == self.max_msg:
                break           
            
            # Pad the error message
            if self.averaged == False:
                error_message = self.pad_along_axis(error_message)
                
            
            # Sequence per task, error, site
            self.error_site_tokens[index, self.codes[error], self.sites[site], counter ] = error_message    
            
    
    def to_dense(self, index_matrix, values):
        
        errors, sites, counts, site_states, error_messages = values
        
        # Loop over the codes and sites
        for i_key in range(len(counts)):
            
            error = errors[i_key]
            site = sites[i_key]
            count = counts[i_key]
            site_state = site_states[i_key]
    
            
            # Fill the counts
            if self.only_msg == False:
                self.fill_counts(index_matrix, error, site, site_state, count)
           
            if self.only_counts == True:
                continue
            
            error_message_sequence = error_messages[i_key]
            
            # Only continue if there exists a message
            if isinstance(error_message_sequence, (list,)):
                
                # Fill the error message
                if self.sequence == True:
                    self.fill_messages_sequence( index_matrix, error, site, error_message_sequence)
                else:
                    self.fill_first_message( index_matrix, error, site, error_message_sequence)
                    

                
    def get_counts_matrix(self, sum_good_bad = False):
        
        self.only_counts = True
        
        self.error_site_counts = np.zeros((self.n_tasks, self.unique_codes, self.unique_sites, 2), dtype=np.int32)
        batch = self.frame
        [self.to_dense(counter, values) for counter, values in enumerate(zip(self.frame['error'], self.frame['site'], 
                                                                             self.frame['count'], self.frame['site_state'],
                                                                             self.frame['msg_encoded'],))]        
        if sum_good_bad == True:
            return self.error_site_counts.sum(axis=3), self.frame[self.label].values
        else:
            return self.error_site_counts, self.frame[self.label].values        
    
    
    def msg_batch(self, start_pos, end_pos):
        
        self.only_counts = False
        
        # Batch of frame
        batch = self.frame.iloc[start_pos : end_pos]
        chunk_size = len(batch)
        
        # Tokens
        if self.averaged == False:
            tokens_key = 'msg_encoded'
            pad_dim = 1
            for messages in batch[tokens_key]:
                for msg in messages:
                    if isinstance(msg, (list,)):
                        if len(msg) > pad_dim:
                            pad_dim = len(msg)
                        
            if pad_dim > self.max_words:
                pad_dim = self.max_words
            self.pad_dim = pad_dim
        else:
            tokens_key = 'avg'
       
        
        #print( self.max_words )
        #print( msg )
        
        # Error site matrix
        self.error_site_counts = np.zeros((chunk_size, self.unique_codes, self.unique_sites, 2), dtype=np.int32)
        
        if self.sequence == True:
            dim = (chunk_size, self.unique_codes, self.unique_sites, self.max_msg, self.pad_dim)
        else:
            dim = (chunk_size, self.unique_codes, self.unique_sites, self.pad_dim)    
        
        
        # Error message matrix
        self.error_site_tokens = np.zeros(dim, dtype=np.int32)
        
        [self.to_dense(counter, values) for counter, values in enumerate(zip(batch['error'], batch['site'], batch['count'],
                                                                          batch['site_state'], batch[tokens_key]))]
        
        if self.only_msg == False:
            #print( np.count_nonzero(self.error_site_tokens) )
            #self.error_site_tokens = np.reshape(
            return [self.error_site_tokens.reshape((chunk_size, self.unique_codes * self.unique_sites, self.pad_dim)) , 
                    self.error_site_counts]   
        else:
            return self.error_site_tokens
    
    
    def gen_batches(self):
        
        for cur_pos in range(0, self.n_tasks, self.batch_size):
 
            next_pos = cur_pos + self.batch_size 
            if next_pos <= self.n_tasks:
                yield (self.msg_batch( cur_pos, next_pos ), self.frame[self.label].iloc[cur_pos : next_pos].values)
            else:
                yield (self.msg_batch( cur_pos, self.n_tasks ), self.frame[self.label].iloc[cur_pos : self.n_tasks].values)   
                  
                    
    def gen_inf_batches(self):
        
        while True:
            try:
                for B in self.gen_batches():
                    yield B
            except StopIteration:
                logging.warning("start over generator loop")
        
