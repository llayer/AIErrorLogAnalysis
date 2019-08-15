import os
import sys
module_path = os.path.abspath(os.path.join('../utils'))
if module_path not in sys.path:
    sys.path.append(module_path)
from create_actionshist_keys import *
from actionshist_utils import *
import numpy as np
import pandas as pd
#from  actionshist_utils import set_labels, get_exit_codes, get_sites
from keras.preprocessing.sequence import pad_sequences

def clean_error_type(error_type):
    
    if len(error_type) > 50:
        return str('TypeError').decode('utf-8')
    else:
        return error_type

def encode_sites_unicode( site ):
    
    if site == 'T3_US_NERSC\xc2\xa0':
        return 'T3_US_NERSC'.decode('utf-8')
    else:
        return site
    

def get_site_info(frame):
    
    non_neg_sites = list(frame[frame['error'] != '-1'.decode('utf-8') ]['site'].unique())
    counts = frame.groupby('site')['count'].sum().to_frame('counts').reset_index()
    frequency = frame['site'].value_counts().to_frame('frequency').reset_index().rename(columns={'index': 'site'})
    
    combined_info = pd.merge(counts, frequency, on=['site'], how = 'outer')  
  
    def neg_sites(site):
        if site in non_neg_sites:
            return False
        else:
            return True
        
    combined_info['only_unknown'] = combined_info['site'].apply(neg_sites)
    return combined_info

def get_error_info(frame):
                   
    counts = frame.groupby('error')['count'].sum().to_frame('counts').reset_index()
    frequency = frame['error'].value_counts().to_frame('frequency').reset_index().rename(columns={'index': 'error'})
    
    combined_info = pd.merge(counts, frequency, on=['error'], how = 'outer')                     
    return combined_info
                   

    
    

def create_input( path_actionshist, path_tokens, max_length = 200, store = True, 
                 store_path = '/eos/user/l/llayer/AIErrorLogAnalysis/data/input/'):
        
    # Load the actionshistory
    print( 'Loading actionshist and generating keys' )
    actionshist = load_data(path_actionshist)
    set_binary_labels(actionshist)
    actionshist_keys = get_keys(actionshist, ignore_neg_code = False)
    
    # Get the error messages
    print( 'Loading tokens' )
    tokens = pd.read_hdf( path_tokens )
    tokens['error_type'] = tokens['error_type'].apply(clean_error_type)
    tokens['avg_w2v'] = tokens['avg_w2v'].apply(lambda x:list( float(entry) for entry in x ))
    tokens.error = tokens.error.astype(str)
    tokens.error = tokens.error.str.encode('utf-8')
    tokens.site = tokens.site.str.encode('utf-8')
    
    
    print( 'Aggregating tokens' )
    # Pad the tokens
    #tokens['error_msg_tokenized'] = list(pad_sequences(tokens['error_msg_tokenized'], maxlen=max_length, 
    #                                         padding="post", truncating="post"))
    # Aggregate the message per task, error, site
    tokens = tokens.groupby(['task_name', 'error', 'site'], as_index=False)['error_msg_tokenized', 'exit_codes',
                                                                            'steps_counter', 'error_type', 
                                                                           'avg_w2v'].agg(lambda x: list(x))
  
             
    print( 'Merging the frames' )
    sparse_frame = pd.merge( actionshist_keys, tokens, on = ['task_name', 'error', 'site'], how='left')
    
    
    # Clean the unicode sites
    sparse_frame['site'] = sparse_frame['site'].apply(encode_sites_unicode)
  
    # Getting the sites and error codes
    site_frame = get_site_info(sparse_frame)
    codes_frame = get_error_info(sparse_frame)
    
    
    print( 'Aggregating the frames' )
    #sparse_frame['error_msg_tokenized'] = sparse_frame['error_msg_tokenized'].replace(np.nan, 0, regex=True)
    sparse_frame = sparse_frame.groupby(['task_name', 'label'], as_index=False)['error', 'site', 'site_state', 'count',
                                                                                'error_msg_tokenized','exit_codes',
                                                                                'steps_counter', 'error_type',
                                                                                'avg_w2v'].agg(lambda x: list(x))
    
        
    if store == True:
        print( 'Storing the frame' )
        
        outfile = store_path + 'input' + '.h5'
        sparse_frame.to_hdf( outfile , 'frame')
        site_frame.to_hdf( outfile , 'frame2')
        codes_frame.to_hdf( outfile, 'frame3')
                  
    return sparse_frame, site_frame, codes_frame
        
        
        
     
    
