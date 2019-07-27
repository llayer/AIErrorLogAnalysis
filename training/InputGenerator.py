import pandas as pd
import json
import itertools
import math
import numpy as np
import ut
from sklearn.model_selection import train_test_split




class InputBatchGenerator(object):
    
    def __init__(self, frame, labels, batch_size, codes, sites, dim_msg, msg = 'tokens'):
        
        self.frame = frame
        self.labels = labels
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
        index_matrix = index % self.batch_size
        
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
        
        return (self.error_site_tokens, self.error_site_counts)       
    
    
    def generate_msg_count_batches(self):
        
        n_tasks = len(self.frame)
        for cur_pos in range(0, n_tasks, self.batch_size):
 
            next_pos = cur_pos + self.batch_size 
            if next_pos <= n_tasks:
                yield ( self.msg_count_batch( cur_pos, next_pos ), self.labels.iloc[cur_pos : next_pos].values )
            else:
                yield ( self.msg_count_batch( cur_pos, n_tasks ), self.labels.iloc[cur_pos : next_pos].values )   
                  
    
    
    def count_matrix(self):
        self.frame.apply(self.build_table_counts, axis=1)
        return self.error_site_counts, self.labels  
    
    
    
    

class InputGenerator(object):
    
    """Class to load the actionshistory.json, the word vectors
        and setup the input matrices
    """
    
    def __init__(self, actionshistory_path):
        
        """
            Stores the actionshistory file and sets the labels and site / error code names
            Params:
              actionshistory_path: path to the actionshistory file
        """
        
        # Load the actionshistory file
        self.actionshistory = pd.read_json(actionshistory_path, orient='index')
        # Reset index
        self.actionshistory = self.actionshistory.reset_index()
        self.actionshistory = self.actionshistory.rename(columns={'index': 'task_name'})
        self.actionshistory['unique_index'] = self.actionshistory.index
        # Get the unique exit codes and sites
        self.good_codes, self.bad_codes = ut.get_exit_codes(self.actionshistory)
        self.good_sites, self.bad_sites = ut.get_sites(self.actionshistory)
        self.unique_sites = list(set(self.good_sites + self.bad_sites)) 
        self.unique_codes = list(set(self.good_codes + self.bad_codes))
        self.unique_codes = sorted(self.unique_codes, key=lambda x: float(x))
        self.sites, self.codes = self.list_to_index(self.unique_sites, self.unique_codes)
        ut.set_binary_labels(self.actionshistory)
    
    
    def inspect_single_task(self, i_task ):
        
        single_task = self.actionshistory.iloc[ i_task ]
        return single_task

    def train_test_split(self, label, split_level):
        self.X_train, self.X_test, self.y_train, self.y_test = \
        train_test_split(self.actionshistory, self.actionshistory[label], test_size = split_level, random_state=0)
    
    
    ##############
    # Functions for the error, site encoding and labeling
    ##############
    
    def prune_error_sites(self, threshold_sites = 0, threshold_errors = 0, ignore_neg_code = False):

        counts_errors, counts_sites = ut.get_zero_sites(self.actionshistory, ignore_neg_code) 
        pruned_errors = counts_errors[counts_errors['counts'] > threshold_errors]
        pruned_sites = counts_sites[counts_sites['counts'] > threshold_sites]

        return pruned_errors['error'].unique(), pruned_sites['site'].unique()
        
    
    def sites_to_tiers(self, sites):
        tiers_to_index = {'T0' : 0, 'T1' : 1, 'T2' : 2, 'T3' : 3}
        sites_tiers = {}
        for site in sites:
            tier = site[0:2].decode('utf8')
            print( tier )
            if tier in tiers_to_index:
                sites_tiers[site] = tiers_to_index[tier]
            else:
                sites_tiers[site] = 4

        return sites_tiers
    
    def set_labels(self):
        
        # Set the labels
        ut.set_labels(self.actionshistory)        
    

    def list_to_index(self, sites, codes):
        
        sites_index = {k: v for v, k in enumerate(sites)}
        codes_index = {k: v for v, k in enumerate(codes)}
        return sites_index, codes_index
    
        
    ##############
    # Functions to set the word vectors / tokens
    ##############
        
        
    def get_input_shape(self):
        
        dim_w2v = self.dim_w2v
        dim_sites = len(self.unique_sites)
        dim_errors = len(self.unique_codes)
        dim_tasks = len(self.actionshistory)
        
        return dim_tasks, dim_errors, dim_sites, dim_w2v + 1
        
        
    def set_w2v_avg(self, path):
        
        # Read the file
        w2v = pd.read_csv(path)
        
        # Convert the word vectors from string back to float
        def str_to_float(self, row):
            # Convert the word vectors from string back to float
            log_msg = row['w2v']
            msg = list(np.float_(log_msg.replace('[','').replace(']', '').split(',')))
            return msg
        w2v['w2v'] = w2v.apply(self.str_to_float, axis=1)
        
        # Create lists with the error, site, message per taskname
        w2v_list = w2v.groupby(['task_name'], as_index=False)['error', 'site', 'w2v'].agg(lambda x: list(x))
        
        # Join the frames on task_name
        self.actionshistory = pd.merge( self.actionshistory, w2v_list, on = ['task_name'], how='left')
        
        # Dimension of the word vectors
        self.dim_msg = len(w2v['w2v'][0])        
        
        
    def set_padded_tokens(self, frame):
        
        # Create lists with the error, site, message per taskname
        tokens_list = frame.groupby(['task_name'], as_index=False)['error', 'site', 'tokens_padded'].agg(lambda x: list(x))
        
        # Join the frames on task_name
        self.actionshistory = pd.merge( self.actionshistory, tokens_list, on = ['task_name'], how='left')
        
        # Dimension of the word vectors
        self.dim_msg = len(frame['tokens_padded'][0])         
        
        
        
    ##############
    # Functions to yield batches for the training
    ##############
    

    def train_generator(self, batch_size):
        
        train_gen = InputBatchGenerator(self.X_train, self.y_train, batch_size, self.codes, self.sites, self.dim_msg)
        
        while True:
            try:
                for B in train_gen.generate_msg_count_batches():
                    yield B
            except StopIteration:
                logging.warning("start over generator loop")          
        
    def test_generator(self, batch_size):
        
        test_gen = InputBatchGenerator(self.X_test, self.y_test, batch_size, self.codes, self.sites, self.dim_msg)
        for B in train_gen.generate_msg_count_batches():
            yield B
   
    
        
    ##############
    # Save the dataframe in chunks
    ##############
        
    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

    
    def store(self, name, chunks = 1):
        
        path = '/bigdata/shared/AIErrorHandling/'
        
        if chunks <= 1:
            data_out = self.actionshistory.drop(['task_name', 'errors', 'parameters'], 1)
            data_out.to_hdf(path + name + '.h5', 'frame')
        
        else:
            size_chunk = int(float(len(self.actionshistory)) / chunks)
            for counter, chunk in enumerate(self.chunker(self.actionshistory, size_chunk)):

                #print counter + 1, '/', chunks
                """
                print 'Start with chunk', counter
                chunk['table'] = chunk.apply(build_table, axis=1)
                print 'Created matrix'
                chunk['table_flattened'] = chunk['table'].apply(lambda x: flatten(x))
                print 'Flattened matrix'
                """
                data_out = chunk.drop(['task_name', 'errors', 'parameters'], 1)
                data_out.to_hdf(path + name + str(counter) + '.h5', 'test')
                #print 'Stored output'
        
        
        
    