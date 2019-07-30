import pandas as pd
import itertools
import math
import numpy as np



class InputBatchGenerator(object):
    
    def __init__(self, frame, label, codes, sites, dim_msg, batch_size = None, msg = 'tokens'):
        
        self.frame = frame
        self.n_tasks = len(frame)
        self.label = label
        self.batch_size = batch_size
        self.codes = codes
        self.sites = sites
        self.dim_msg = dim_msg
        self.msg = msg
    
    
    def fill_counts( self, index_matrix, sites, site_state ):
        
        """
        Fill the matrix with counts
        """
        
        if len(sites.keys()) == 0 or len(sites.values()) == 0:
            return 

        for exit_code, site_dict in sites.items():
            exit_code = exit_code.encode("utf-8")
            for site, count in site_dict.items():
        
                if site == 'NoReportedSite':
                    continue
               
                site = site.encode('utf-8')
                self.error_site_counts[index_matrix, self.codes[exit_code], self.sites[site], site_state] = count   
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
        
        # Add word vectors
        if isinstance(log_sites, (list,)):

            for i in range(len(log_sites)):
                if log_sites[i] == 'NoReportedSite':
                    continue
                                    
                self.error_site_tokens[index_matrix, self.codes[log_errors[i]], self.sites[log_sites[i]]] = log_msg[i]
                
           
    def msg_count_batch(self, start_pos, end_pos):
        
        self.error_site_tokens = np.zeros((self.batch_size, len(self.codes), len(self.sites), self.dim_msg))
        self.error_site_counts = np.zeros((self.batch_size, len(self.codes), len(self.sites), 2))
        self.frame.iloc[start_pos : end_pos].apply(self.build_table_msg, axis=1)
        self.frame.iloc[start_pos : end_pos].apply(self.build_table_msg, axis=1)
        
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
            
    
    
    def count_matrix(self):
        
        self.error_site_counts = np.zeros((self.n_tasks, len(self.codes), len(self.sites), 2))
        self.frame.apply(self.build_table_counts, axis=1)
        return self.error_site_counts, self.frame[self.label]  
    
    
    
    
    