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


def create_input( path_actionshist, path_tokens, mode = 'tokens', max_length = 200, store = False, 
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
        # Pad the tokens
        tokens['error_msg_tokenized'] = list(pad_sequences(tokens['error_msg_tokenized'], maxlen=max_length, 
                                                 padding="post", truncating="post"))
        # Aggregate the message per task, error, site
        tokens = tokens.groupby(['task_name', 'error', 'site'], as_index=False)['error_msg_tokenized'].agg(lambda x: list(x))
        
        
    else:
        tokens = tokens.groupby(['task_name', 'error', 'site'], as_index=False)['avg_w2v'].agg(lambda x: list(x))
        
    print( 'Merging the frames' )
    sparse_frame = pd.merge( actionshist_keys, tokens, on = ['task_name', 'error', 'site'], how='left')
    
    print( 'Aggregating the frames' )
    if mode == 'index':
        sparse_frame = sparse_frame.groupby(['task_name', 'label'], as_index=False)['error', 'site', 'site_state', 'count',
                                                                                    'error_msg_tokenized'].agg(lambda x: list(x))
    else:
        sparse_frame = sparse_frame.groupby(['task_name', 'label'], as_index=False)['error', 'site', 'site_state',
                                                                                    'count', 'avg_w2v'].agg(lambda x: list(x))

        
    if store == True:
        print( 'Storing the frame' )
        sparse_frame.to_hdf(store_path + mode + 'h5', 'frame')
                  
    return sparse_frame
        
        
        
     
    