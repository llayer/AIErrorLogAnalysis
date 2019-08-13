import pandas as pd
import itertools
import math
import numpy as np
from keras.utils import to_categorical



class InputBatchGenerator(object):
    
    def __init__(self, frame, label, codes, sites, pad_dim, batch_size = 1, max_msg = 5, max_msg_per_error = 50, 
                 max_msg_per_wf = 200, mode = 'default', only_counts = False, averaged = False):
        
        self.frame = frame
        self.frame['unique_index'] = self.frame.reset_index().index
        self.n_tasks = len(frame)
        self.label = label
        self.batch_size = batch_size
        self.codes = codes
        self.sites = sites
        self.pad_dim = pad_dim
        self.max_msg = max_msg
        self.max_msg_per_error = max_msg_per_error
        self.max_msg_per_wf = max_msg_per_wf
        self.mode = mode
        self.averaged = averaged
        self.only_counts = only_counts
        self.unique_sites = len(list(set(self.sites.values())))
        self.unique_codes = len(list(set(self.codes.values())))
        self.n_tasks = len(frame)
       
    
    

    def pad_along_axis(self, array, axis=0):

        array = np.array(array)
        pad_size = self.pad_dim - array.shape[axis]
        axis_nb = len(array.shape)

        if pad_size < 0:
            return array[0:self.pad_dim]

        npad = [(0, 0) for x in range(axis_nb)]
        npad[axis] = (0, pad_size)

        b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

        return b
    
    
    def fill_counts(self, index, error, site, site_state, count):
        
        # Encode good and bad sites
        if site_state == 'good':
            site_state_encoded = 0
        else:
            site_state_encoded = 1

        self.error_site_counts[index, self.codes[error], self.sites[site], site_state_encoded] += count
                
 
    def fill_messages(self, index, error, site, error_message_sequence, i_key, used_codes):
        
        # Loop over the error message sequence
        for counter, error_message in enumerate(error_message_sequence):

            
            # Pad the error message
            error_message = self.pad_along_axis(error_message)
            
            #print (index, error, site, i_key, counter)
            #print (error_message)
            
            # Stop when maximal message is reached
            if counter == self.max_msg:
                break             
            
            # Sequence per task, error, site
            if self.mode == 'default':               
                self.error_site_tokens[index, self.codes[error], self.sites[site], counter ] = error_message
            
            # Sequence per task, error
            elif self.mode == 'sum_sites':
                if codes_used[self.codes[error]] == self.max_msg_per_error:
                    break
                self.error_site_tokens[index, self.codes[error], codes_used[self.codes[error]], counter] = error_message
                
            # Sequence per task
            elif self.mode == 'sum_sites_errors':
                if i_key == self.max_msg_per_wf:
                    break
                self.error_site_tokens[index, i_key, counter] = error_message

            else:
                print( 'Error' )       
            
    
    def to_dense(self, index_matrix, values):

        errors, sites, counts, site_states, error_messages = values

        # Store the used codes
        codes_used = {}
        
        # Loop over the codes and sites
        for i_key in range(len(counts)):
            
            error = errors[i_key]
            site = sites[i_key]
            count = counts[i_key]
            site_state = site_states[i_key]
            
            # Fill the counts
            self.fill_counts(index_matrix, error, site, site_state, count)
            
            if self.only_counts == True:
                continue
            
            error_message_sequence = error_messages[i_key]
            
            # Only continue if there exists a message
            if isinstance(error_message_sequence, (list,)):
                
                # Store the used codes
                if self.codes[error] in codes_used:
                    codes_used[ self.codes[error] ] += 1
                else:
                    codes_used[ self.codes[error] ] = 0
                
                # Fill the error message
                self.fill_messages( index_matrix, error, site, error_message_sequence, i_key, codes_used)

                
    def get_counts_matrix(self, sum_good_bad = False):
        
        self.only_counts = True
        
        self.error_site_counts = np.zeros((self.n_tasks, self.unique_codes, self.unique_sites, 2), dtype=np.float32)
        batch = self.frame
        [self.to_dense(counter, values) for counter, values in enumerate(zip(self.frame['error'], self.frame['site'], 
                                                                             self.frame['count'], self.frame['site_state'],
                                                                             self.frame['error_msg_tokenized']))]        
        if sum_good_bad == True:
            return self.error_site_counts.sum(axis=3), self.frame[self.label].values
        else:
            return self.error_site_counts, self.frame[self.label].values        
    
    
    def msg_batch(self, start_pos, end_pos):
        
        self.only_counts = False
        
        # Error site matrix
        self.error_site_counts = np.zeros((self.batch_size, self.unique_codes, self.unique_sites, 2))
        
        # Error message matrix
        if self.mode == 'default':
            self.error_site_tokens = np.zeros((self.batch_size, self.unique_codes, self.unique_sites, 
                                               self.max_msg, self.pad_dim))
        elif self.mode == 'sum_sites':
            self.error_site_tokens = np.zeros((self.batch_size, self.unique_codes, self.max_msg_per_error, 
                                               self.max_msg, self.pad_dim))
        elif self.mode == 'sum_sites_errors':
            self.error_site_tokens = np.zeros((self.batch_size, self.max_msg_per_wf, self.max_msg, self.pad_dim))
        else:
            print( 'No valid configuration chosen' )
        
        batch = self.frame.iloc[start_pos : end_pos]
        [self.to_dense(counter, values) for counter, values in enumerate(zip(batch['error'], batch['site'], batch['count'],
                                                                          batch['site_state'], batch['error_msg_tokenized']))]
        
        return [self.error_site_tokens, self.error_site_counts]   
    
    
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
        
        
                
                