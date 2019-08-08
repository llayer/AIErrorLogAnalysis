import pandas as pd
import itertools
import math
import numpy as np
from keras.utils import to_categorical


class InputBatchGenerator(object):
    
    def __init__(self, frame, label, codes, sites, dim_msg, max_msg, 
                 mode = 'default', batch_size = None, msg = 'tokens'):
        
        self.frame = frame
        self.frame['unique_index'] = self.frame.reset_index().index
        self.n_tasks = len(frame)
        self.label = label
        self.batch_size = batch_size
        self.codes = codes
        self.sites = sites
        self.dim_msg = dim_msg
        self.msg = msg
        self.mode = mode
        self.max_msg = max_msg
        self.max_msg_per_wf, self.max_msg_per_site = self.get_max_msg()
        
    
    def get_max_msg( self ):
        
        binary_msg = self.binary_msg_matrix()
        return int(np.max(binary_msg.sum(axis=2).sum(axis=1))), int(np.max(binary_msg.sum(axis=2)))    
    
    
    def fill_counts( self, index_matrix, sites, site_state ):
        
        """
        Fill the matrix with counts
        """
        
        if len(sites.keys()) == 0 or len(sites.values()) == 0:
            return 

        for exit_code, site_dict in sites.items():
            exit_code = exit_code.encode("utf-8")
            for site, count in site_dict.items():
        
                #if site == 'NoReportedSite':
                #    continue
               
                site = site.encode('utf-8')
                self.error_site_counts[index_matrix, self.codes[exit_code], self.sites[site], site_state] += count   
                #print (index_matrix, self.codes[exit_code], self.sites[site], count , site_state)

                
    def build_table_counts(self, row):
        
        """
        Fill the matrix batch with counts of good and bad sites
        """
        
        errors = row['errors']
        index = row['unique_index']
        
        if self.batch_size is not None:
            index_matrix = index % self.batch_size
        else:
            index_matrix = index
        
        sites_good = errors['good_sites'] 
        sites_bad = errors['bad_sites']
        
        self.fill_counts(index_matrix, sites_good, 0)
        self.fill_counts(index_matrix, sites_bad, 1)    


    def build_table_msg(self, row):

        """
        Fill the matrix batch with the error messages
        """
        
        # Build the site-error-w2v matrix table
        log_sites = row['site']
        log_errors = row['error']
        
        if self.msg == 'tokens':
            log_msg = row['tokens_padded']
        elif self.msg == 'w2v_avg':
            log_msg = row['w2v']
            
        index = row['unique_index']
        index_matrix = index % self.batch_size
        
        codes_used = {}
        
        # Check that there is at least one message
        if isinstance(log_sites, (list,)):
            
            print( len(log_sites) )
            print( index_matrix, index )
            # Loop over site
            for i in range(len(log_sites)):
                
                # Loop over messages per wf, site, error
                for j in range(len(log_msg[i])):
                    
                    #self.error_site_tokens[index_matrix, self.codes[log_errors[i]], 
                    #                       self.sites[log_sites[i]], j ] = log_msg[i][j] 

                    if self.mode == 'default':               
                        self.error_site_tokens[index_matrix, self.codes[log_errors[i]], 
                                               self.sites[log_sites[i]], j ] = log_msg[i][j]
                        
                    elif self.mode == 'sum_sites':
                        if self.codes[log_errors[i]] in codes_used:
                            codes_used[ self.codes[log_errors[i]] ] += 1
                        else:
                            codes_used[ self.codes[log_errors[i]] ] = 0
                        self.error_site_tokens[index_matrix, self.codes[log_errors[i]], 
                                               codes_used[self.codes[log_errors[i]]], j] = log_msg[i][j]
                        
                    elif self.mode == 'sum_sites_errors':
                        self.error_site_tokens[index_matrix, i, j] = log_msg[i][j]
                    
                    else:
                        print( 'Error' )
                   
                    if j+1 == self.max_msg:
                        break
                
           
    def msg_count_batch(self, start_pos, end_pos):
        
        self.error_site_counts = np.zeros((self.batch_size, len(self.codes), len(self.sites), 2))
        #self.error_site_tokens = np.zeros((self.batch_size, len(self.codes), len(self.sites), self.max_msg, self.dim_msg))
        
        print( self.mode )
        if self.mode == 'default':
            self.error_site_tokens = np.zeros((self.batch_size, len(self.codes), len(self.sites), self.max_msg, self.dim_msg))
        elif self.mode == 'sum_sites':
            self.error_site_tokens = np.zeros((self.batch_size, len(self.codes), self.max_msg_per_site, 
                                               self.max_msg, self.dim_msg))
        elif self.mode == 'sum_sites_errors':
            self.error_site_tokens = np.zeros((self.batch_size, self.max_msg_per_wf, self.max_msg, self.dim_msg))
        else:
            print( 'No valid configuration chosen' )
        
        
        self.frame.iloc[start_pos : end_pos].apply(self.build_table_msg, axis=1)
        self.frame.iloc[start_pos : end_pos].apply(self.build_table_counts, axis=1)
        
        return [self.error_site_tokens, self.error_site_counts]     
    
    
    def gen_msg_count_batches(self):
        
        for cur_pos in range(0, self.n_tasks, self.batch_size):
 
            next_pos = cur_pos + self.batch_size 
            if next_pos <= self.n_tasks:
                yield ( self.msg_count_batch( cur_pos, next_pos ), self.frame[self.label].iloc[cur_pos : next_pos].values )
            else:
                yield ( self.msg_count_batch( cur_pos, n_tasks ), self.frame[self.label].iloc[cur_pos : n_tasks].values )   
                  
    def gen_inf_count_msg_batches(self):
        
        while True:
            try:
                for B in self.gen_msg_count_batches():
                    yield B
            except StopIteration:
                logging.warning("start over generator loop")
           
        
    def build_table_msg_bin(self, row):

        """
        Fill the matrix batch with the error messages
        """
        
        # Build the site-error-w2v matrix table
        log_sites = row['site']
        log_errors = row['error']
        
        if self.msg == 'tokens':
            log_msg = row['tokens_padded']
        elif self.msg == 'w2v_avg':
            log_msg = row['w2v']
            
        index_matrix = row['unique_index']
        
        # Add word vectors
        if isinstance(log_sites, (list,)):

            for i in range(len(log_sites)):
                                    
                self.error_site_tokens_bin[index_matrix, self.codes[log_errors[i]], self.sites[log_sites[i]]] = 1  
               
    
    def binary_msg_matrix(self):
        
        self.error_site_tokens_bin = np.zeros((self.n_tasks, len(self.codes), len(self.sites)))
        self.frame.apply(self.build_table_msg_bin, axis=1)
        return self.error_site_tokens_bin
                                          
    
    def count_matrix(self, sum_good_bad = False):
        
        n_sites = len(list(set(self.sites.values())))
        n_codes = len(list(set(self.codes.values())))
        
        self.error_site_counts = np.zeros((self.n_tasks, n_codes, n_sites, 2))
        self.frame.apply(self.build_table_counts, axis=1)
       
        if sum_good_bad == True:
            return self.error_site_counts.sum(axis=3), self.frame[self.label].values
        else:
            return self.error_site_counts, self.frame[self.label].values #to_categorical(self.frame[self.label])  
    
    
    
    
    