import pandas as pd
import itertools
import math
import numpy as np
from keras.utils import to_categorical



class InputBatchGenerator(object):
    
    def __init__(self, frame, label, codes, sites, pad_dim, batch_size = 1, max_msg = 5, max_msg_per_error = 50, 
                 max_msg_per_wf = 200, mode = 'default', only_counts = False, averaged = False, first_only = False,
                 count_msg = True):
        
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
        self.first_only = first_only
        self.only_counts = only_counts
        self.count_msg = count_msg
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
    
    
    def fill_first_message(self, index, error, site, error_message_sequence, i_key, used_codes, exit_code):
        
        # Loop over the error message sequence
        for counter, error_message in enumerate(error_message_sequence):
            
            # Stop when maximal message is reached
            if error != exit_code:
                        
                # Pad the error message
                if self.averaged == False:
                    error_message = self.pad_along_axis(error_message)

                # Sequence per task, error, site
                if self.mode == 'default':               
                    #print( self.codes[error], error, self.sites[site], site, counter )
                    self.error_site_tokens[index, self.codes[error], self.sites[site]] = error_message

                # Sequence per task, error
                elif self.mode == 'sum_sites':
                    if codes_used[self.codes[error]] == self.max_msg_per_error:
                        break
                    self.error_site_tokens[index, self.codes[error], codes_used[self.codes[error]]] = error_message

                # Sequence per task
                elif self.mode == 'sum_sites_errors':
                    if i_key == self.max_msg_per_wf:
                        break
                    self.error_site_tokens[index, i_key] = error_message

                else:
                    print( 'Error' )   

                # Stop after first message equal to error code
                break   
    
 
    def fill_messages_sequence(self, index, error, site, error_message_sequence, i_key, used_codes):
        
        # Loop over the error message sequence
        for counter, error_message in enumerate(error_message_sequence):
    
            
            # Stop when maximal message is reached
            if counter == self.max_msg:
                break           
            
            # Pad the error message
            if self.averaged == False:
                error_message = self.pad_along_axis(error_message)
            
            # Sequence per task, error, site
            if self.mode == 'default':               
                #print( self.codes[error], error, self.sites[site], site, counter )
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

        errors, sites, counts, site_states, error_messages, exit_codes = values

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
            exit_code = exit_codes[i_key]
            
            # Only continue if there exists a message
            if isinstance(error_message_sequence, (list,)):
                
                # Store the used codes
                if self.codes[error] in codes_used:
                    codes_used[ self.codes[error] ] += 1
                else:
                    codes_used[ self.codes[error] ] = 0
                
                # Fill the error message
                if self.first_only == False:
                    self.fill_messages_sequence( index_matrix, error, site, error_message_sequence, i_key, codes_used)
                else:
                    self.fill_first_message( index_matrix, error, site, error_message_sequence, i_key, codes_used, exit_code)

                
    def get_counts_matrix(self, sum_good_bad = False):
        
        self.only_counts = True
        
        self.error_site_counts = np.zeros((self.n_tasks, self.unique_codes, self.unique_sites, 2), dtype=np.float32)
        batch = self.frame
        [self.to_dense(counter, values) for counter, values in enumerate(zip(self.frame['error'], self.frame['site'], 
                                                                             self.frame['count'], self.frame['site_state'],
                                                                             self.frame['error_msg_tokenized'], 
                                                                             self.frame['exit_codes']))]        
        if sum_good_bad == True:
            return self.error_site_counts.sum(axis=3), self.frame[self.label].values
        else:
            return self.error_site_counts, self.frame[self.label].values        
    
    
    def msg_batch(self, start_pos, end_pos):
        
        self.only_counts = False
        
        # Batch of frame
        batch = self.frame.iloc[start_pos : end_pos]
        chunk_size = len(batch)
        
        # Error site matrix
        self.error_site_counts = np.zeros((chunk_size, self.unique_codes, self.unique_sites, 2))
        
        if self.mode == 'default':
            if self.first_only == False:
                dim = (chunk_size, self.unique_codes, self.unique_sites, self.max_msg, self.pad_dim)
            else:
                dim = (chunk_size, self.unique_codes, self.unique_sites, self.pad_dim)
        elif self.mode == 'sum_sites':
            if self.first_only == False:
                dim = (chunk_size, self.unique_codes, self.max_msg_per_error, self.max_msg, self.pad_dim)
            else:
                dim = (chunk_size, self.unique_codes, self.max_msg_per_error, self.pad_dim)           
        elif self.mode == 'sum_sites_errors':
            if self.first_only == False:
                dim = (chunk_size, self.max_msg_per_wf, self.max_msg, self.pad_dim)
            else:
                dim = (chunk_size, self.max_msg_per_wf, self.pad_dim)
        else:
            print( 'No valid configuration chosen' )        
        
        
        # Error message matrix
        self.error_site_tokens = np.zeros(dim)
        
        # Tokens
        if self.averaged == False:
            tokens_key = 'error_msg_tokenized'
        else:
            tokens_key = 'avg_w2v'
        
        [self.to_dense(counter, values) for counter, values in enumerate(zip(batch['error'], batch['site'], batch['count'],
                                                                          batch['site_state'], batch[tokens_key], 
                                                                          batch['exit_codes']))]
        
        if self.count_msg == True:
            return [self.error_site_tokens, self.error_site_counts]   
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
        
        
                
                