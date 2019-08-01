import pandas as pd
import json
import itertools
import math
import numpy as np
import ut
from keras.preprocessing.sequence import pad_sequences



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
    
    
    ##############
    # Functions for the error, site encoding and labeling
    ##############
    
    def prune_error_sites(self, threshold_sites = 0, threshold_errors = 0, ignore_neg_code = False):

        counts_errors, counts_sites, n_keys = ut.get_zero_sites(self.actionshistory, ignore_neg_code) 
        pruned_errors = counts_errors[counts_errors['counts'] > threshold_errors]
        pruned_sites = counts_sites[counts_sites['counts'] > threshold_sites]

        return pruned_errors['error'].unique(), pruned_sites['site'].unique(), n_keys
        
    
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
        
        dim_msg = self.dim_msg
        dim_sites = len(self.unique_sites)
        dim_errors = len(self.unique_codes)
        dim_tasks = len(self.actionshistory)
        
        return dim_tasks, dim_errors, dim_sites, dim_msg
        
        
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
        
        
    def set_padded_tokens(self, path = '/nfshome/llayer/data/w2v_messages.h5', max_length = 200):
        
        tokens = pd.read_hdf(path)
        
        tokens.error = tokens.error.astype(str)
        tokens.error = tokens.error.str.encode('utf-8')
        tokens.site = tokens.site.str.encode('utf-8')
        
        tokens['tokens_padded'] = list(pad_sequences(tokens['encoded_tokens'], maxlen=max_length, padding="post", truncating="post"))
        
        # Create lists with the error, site, message per taskname
        tokens_list = tokens.groupby(['task_name'], as_index=False)['error', 'site', 'tokens_padded'].agg(lambda x: list(x))
        
        # Join the frames on task_name
        self.actionshistory = pd.merge( self.actionshistory, tokens_list, on = ['task_name'], how='left')
        
        # Dimension of the word vectors
        self.dim_msg = len(tokens['tokens_padded'][0])         
   
    
        
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
        
        
        
    