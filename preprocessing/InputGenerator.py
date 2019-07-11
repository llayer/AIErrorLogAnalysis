import pandas as pd
import json
import itertools
import math
import numpy as np
import create_data


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
        self.good_codes, self.bad_codes = create_data.get_exit_codes(self.actionshistory)
        self.good_sites, self.bad_sites = create_data.get_sites(self.actionshistory)
        self.unique_sites = list(set(self.good_sites + self.bad_sites)) 
        self.unique_codes = list(set(self.good_codes + self.bad_codes))
        self.unique_codes = sorted(self.unique_codes, key=lambda x: float(x))


    def set_labels(self):
        
        # Set the labels
        create_data.set_labels(self.actionshistory)        
    
    def build_table(self, errors, sites, codes):

        error_site = np.zeros((len(codes), len(sites)))

        if len(errors.keys()) == 0 or len(errors.values()) == 0:
            return error_site

        else:
            for exit_code, site_dict in zip(errors.keys(), errors.values()):
                for site, count in site_dict.items():
                    exit_code = exit_code.encode('utf-8')
                    site = site.encode('utf-8')
                    error_site[codes[exit_code], sites[site]] = 0 if math.isnan(count) else count

            return error_site

        
    def list_to_index(self, sites, codes):
        
        sites_index = {k: v for v, k in enumerate(sites)}
        codes_index = {k: v for v, k in enumerate(codes)}
        return sites_index, codes_index
    
    
    def generate_error_site_matrix(self, site):
        
        if site == 'good':
            
            sites_index, codes_index = self.list_to_index(self.good_sites, self.good_codes)
            self.actionshistory['table_good_sites'] = \
            self.actionshistory['errors'].apply(lambda x: self.build_table(x['good_sites'], sites_index, codes_index ))
            
        if site == 'bad':
            
            sites_index, codes_index = self.list_to_index(self.bad_sites, self.bad_codes)
            self.actionshistory['table_bad_sites'] = \
            self.actionshistory['errors'].apply(lambda x: self.build_table(x['bad_sites'], sites_index, codes_index ))
            
        if site == 'merged':
            
            sites_index, codes_index = self.list_to_index(self.unique_sites, self.unique_codes)
            self.actionshistory['table_good_sites_unique'] = \
            self.actionshistory['errors'].apply(lambda x: self.build_table(x['good_sites'], sites_index, codes_index)) 
            self.actionshistory['table_bad_sites_unique'] = \
            self.actionshistory['errors'].apply(lambda x: self.build_table(x['bad_sites'], sites_index, codes_index)) 
            self.actionshistory['table_unique'] = \
            np.add(self.actionshistory['table_good_sites_unique'], self.actionshistory['table_bad_sites_unique'])

            
    def inspect_single_task(self, i_task ):
        
        single_task = self.actionshistory.iloc[ i_task ]
        return single_task
        
    
    def str_to_float(self, row):
        
        # Convert the word vectors from string back to float
        log_msg = row['w2v']
        msg = list(np.float_(log_msg.replace('[','').replace(']', '').split(',')))
        return msg

    
    def build_table_w2v(self, row):

        # Build the site-error-w2v matrix table
        errors = row['errors']
        sites_good = errors['good_sites'] 
        sites_bad = errors['bad_sites']
        log_sites = row['site']
        log_errors = row['error']
        log_msg = row['w2v']

        sites, codes = self.list_to_index(self.unique_sites, self.unique_codes)
        
        error_site_w2v = np.zeros((len(codes), len(sites), (self.dim_w2v + 1)))

        # Add exit code
        # Good sites
        for exit_code, site_dict in zip(sites_good.keys(), sites_good.values()):
            for site, count in site_dict.items():
                exit_code = exit_code.encode('utf-8')
                site = site.encode('utf-8')
                error_site_w2v[codes[exit_code], sites[site], 0] = 0 if math.isnan(count) else count
        # Bad sites
        for exit_code, site_dict in zip(sites_bad.keys(), sites_bad.values()):
            for site, count in site_dict.items():
                exit_code = exit_code.encode('utf-8')
                site = site.encode('utf-8')    
                error_site_w2v[codes[exit_code], sites[site], 0] = 0 if math.isnan(count) else count
                
                
        # Add word vectors
        if isinstance(log_sites, (list,)):

            for i in range(len(log_sites)):
                if log_sites[i] == 'NoReportedSite':
                    continue
                error_site_w2v[codes[str(log_errors[i])], sites[str(log_sites[i])], 1:] = log_msg[i]

        return error_site_w2v
    
    
    def generate_error_site_w2v_matrix(self, w2v_path):
        
        # Read the file
        w2v = pd.read_csv(w2v_path)
        # Convert the word vectors from string back to float
        w2v['w2v'] = w2v.apply(self.str_to_float, axis=1)
        
        # Create lists with the error, site, message per taskname
        w2v_list = w2v.groupby(['task_name'], as_index=False)['error', 'site', 'w2v'].agg(lambda x: list(x))
        
        # Join the frames on task_name
        self.actionshistory = pd.merge( self.actionshistory, w2v_list, on = ['task_name'], how='left')
        
        # Dimension of the word vectors
        self.dim_w2v = len(w2v['w2v'][0])
        
        # Set up the matrix - at the moment only merged
        self.actionshistory['table_w2v'] = self.actionshistory.apply(self.build_table_w2v, axis=1)


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

                print counter + 1, '/', chunks
                """
                print 'Start with chunk', counter
                chunk['table'] = chunk.apply(build_table, axis=1)
                print 'Created matrix'
                chunk['table_flattened'] = chunk['table'].apply(lambda x: flatten(x))
                print 'Flattened matrix'
                """
                data_out = chunk.drop(['task_name', 'errors', 'parameters'], 1)
                data_out.to_hdf(path + name + str(counter) + '.h5', 'test')
                print 'Stored output'
        
        
        
    