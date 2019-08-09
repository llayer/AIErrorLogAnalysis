import os
import sys
module_path = os.path.abspath(os.path.join('../utils'))
if module_path not in sys.path:
    sys.path.append(module_path)
from create_actionshist_keys import *
from actionshist_utils import *

import pandas as pd
#from  actionshist_utils import set_labels, get_exit_codes, get_sites
from keras.preprocessing.sequence import pad_sequences


def get_site_info(frame):
    
    non_neg_sites = list(frame[frame['error'] != '-1'.encode('utf-8') ]['site'].unique())
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
                   

    
    

def create_input( path_actionshist, path_tokens, mode = 'tokens', max_length = 200, store = True, 
                 store_path = '/eos/user/l/llayer/AIErrorLogAnalysis/data/input/'):
        
    # Load the actionshistory
    print( 'Loading actionshist and generating keys' )
    actionshist = load_data(path_actionshist)
    set_binary_labels(actionshist)
    actionshist_keys = get_keys(actionshist, ignore_neg_code = False)
    
    # Get the error messages
    print( 'Loading tokens' )
    tokens = pd.read_hdf( path_tokens )
    tokens.error = tokens.error.astype(str)
    tokens.error = tokens.error.str.encode('utf-8')
    tokens.site = tokens.site.str.encode('utf-8')
    
    
    if mode == 'tokens':
        print( 'Aggregating tokens' )
        # Pad the tokens
        tokens['error_msg_tokenized'] = list(pad_sequences(tokens['error_msg_tokenized'], maxlen=max_length, 
                                                 padding="post", truncating="post"))
        # Aggregate the message per task, error, site
        tokens = tokens.groupby(['task_name', 'error', 'site'], as_index=False)['error_msg_tokenized'].agg(lambda x: list(x))
        
        
    else:
        print( 'Aggregating w2v average' )
        tokens = tokens.groupby(['task_name', 'error', 'site'], as_index=False)['avg_w2v'].agg(lambda x: list(x))
        
    print( 'Merging the frames' )
    sparse_frame = pd.merge( actionshist_keys, tokens, on = ['task_name', 'error', 'site'], how='left')
    
    # Getting the sites and error codes
    site_frame = get_site_info(sparse_frame)
    codes_frame = get_error_info(sparse_frame)
    #unique_sites = sparse_frame['site'].unique()
    #unique_codes = sparse_frame['error'].unique()
    #site_frame = pd.DataFrame(unique_sites, columns = ['site'])
    #codes_frame = pd.DataFrame(unique_codes, columns = ['error_code'])
    
    print( 'Aggregating the frames' )
    if mode == 'tokens':
        sparse_frame = sparse_frame.groupby(['task_name', 'label'], as_index=False)['error', 'site', 'site_state', 'count',
                                                                                    'error_msg_tokenized'].agg(lambda x: list(x))
    else:
        sparse_frame = sparse_frame.groupby(['task_name', 'label'], as_index=False)['error', 'site', 'site_state',
                                                                                    'count', 'avg_w2v'].agg(lambda x: list(x))

        
    if store == True:
        print( 'Storing the frame' )
        
        outfile = store_path + 'input_' + mode + '.h5'
        sparse_frame.to_hdf( outfile , 'frame')
        site_frame.to_hdf( outfile , 'frame2')
        codes_frame.to_hdf( outfile, 'frame3')
                  
    return sparse_frame, site_frame, codes_frame
        
        
        
     
    