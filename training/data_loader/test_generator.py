import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('../../utils'))
if module_path not in sys.path:
    sys.path.append(module_path)
from create_actionshist_keys import *
from actionshist_utils import *
import index
import input_batch_generator as gen
import random

def load_token_counts(path_tokens = '/eos/user/l/llayer/AIErrorLogAnalysis/data/word2vec/tokens_index_avg.h5'):
    
    tokens = pd.read_hdf( path_tokens )
    tokens = tokens.groupby(['task_name', 'error', 'site'], as_index=False)['exit_codes', 'steps_counter',
                                                                                'error_type'].agg(lambda x: list(x))
    total_tokens = len(tokens)
    del tokens
    return total_tokens


def load_actionshist_counts(actionshist):
    
    actionshist_keys = get_keys(actionshist, ignore_neg_code = False)
    total_counts = actionshist_keys['count'].sum()
    total_keys = len(actionshist_keys['count'])
    
    del actionshist_keys
    return total_counts, total_keys


def count_matrix_sum(gen, total_counts, total_keys = None):
        
    X, _ = gen.get_counts_matrix()
    binary = X > 0
    
    assert(total_counts == X.sum())
    if total_keys is not None:
        assert(total_keys == binary.sum())
        del binary
    
    del X
    
    return True


def t_count_matrix(input_ml, codes, sites, total_counts, total_keys):
    
    
    # Standard count matrix
    padding_size = 200
    sites_index, codes_index = index.to_index(list(sites['site']), list(codes['error']))
    generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index,
                                                      padding_size, batch_size=10)
    count_matrix_sum(generator, total_counts, total_keys)
    print( 'Pass standard test' )
    
    # Prune without counts
    codes_index, sites_index = index.prune_to_index(codes, sites, only_unknown = True,
                                                    counts = False, error_threshold = 100, site_threshold = 1000)
    generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index,
                                                      padding_size, batch_size=10)    
    count_matrix_sum(generator, total_counts)
    print( 'Pass prune unweighted test' )
    
    # Prune with counts
    codes_index, sites_index = index.prune_to_index(codes, sites, only_unknown = True, counts = True, 
                                                    error_threshold = 1000, site_threshold = 10000)
    generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index,
                                                      padding_size, batch_size=10) 
    count_matrix_sum(generator, total_counts)
    print( 'Pass prune weighted test' )
    
    # Prune to tiers:
    sites_index, codes_index = index.to_index(list(sites['site']), list(codes['error']))
    sites_index = index.tiers_to_index( list(sites['site']) )
    assert(len(set(sites_index.values())) == 5)
    generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index,
                                                      padding_size, batch_size=10) 
    count_matrix_sum(generator, total_counts)
    print( 'Pass tier test' )
    


def t_batch_gen(gen, total_counts, msg_counts):
        
    sum_counts = []
    sum_msg = []
    
    for X,y in gen.gen_batches():
        
        n_msgs = X[0].sum(axis=4).sum(axis=3) > 0
        sum_msg.append(n_msgs.sum())
        sum_counts.append(X[1].sum())
        
    #print np.array(sum_counts).sum()
    #print np.array(sum_msg).sum()
    assert( np.array(sum_counts).sum() == total_counts )
    assert( np.array(sum_msg).sum() == msg_counts)



def get_site(sites):
    indices = []
    for error, site_dict in sites.items():
        error = error.encode("utf-8")
        for site, count in site_dict.items():
            site = site.encode("utf-8")
            indices.append(( error, site, count ) )
    return indices


def count_indices( task ):
    errors = task['errors']
    good = errors['good_sites']
    bad = errors['bad_sites']
    return get_site(good), get_site(bad)


def setup_count_matrix( good, bad, codes, sites ):
    
    
    matrix = np.zeros((len(codes), len(sites), 2))
    
    for entry in good:
        matrix[ codes[entry[0]], sites[entry[1]], 0 ] = entry[2]
    for entry in bad:
        matrix[ codes[entry[0]], sites[entry[1]], 1 ] = entry[2]    
        
    return matrix


def setup_msg_matrix(codes, sites, errors, site, error_message_sequences, max_msg, pad_dim):
    
    print 'New'
    print errors, sites, error_message_sequences
    
    def pad_along_axis(array, pad_dim, axis=0):

        array = np.array(array)
        pad_size = pad_dim - array.shape[axis]
        axis_nb = len(array.shape)

        if pad_size < 0:
            return array[0: pad_dim]

        npad = [(0, 0) for x in range(axis_nb)]
        npad[axis] = (0, pad_size)

        b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

        return b
    
    
    unique_sites = len(list(set(sites.values())))
    unique_codes = len(list(set(codes.values())))
    error_site_tokens = np.zeros((1, unique_codes, unique_sites, max_msg, pad_dim))
    
    for error, site, error_message_sequence in zip( errors, sites, error_message_sequences ):
        
        if isinstance(error_message_sequence, (list,)):
            print error, site, error_message_sequence
            # Loop over the error message sequence
            for counter, error_message in enumerate(error_message_sequence):

            
       
                # Pad the error message
                error_message = pad_along_axis(error_message, pad_dim)
                if counter == max_msg:
                    break    
                #print type(error_message)
                try:
                    error_site_tokens[1, codes[error], sites[site], counter ] = error_message
                except:
                    pass #print codes, sites, error, site, counter, error_message
                
    return error_site_tokens


def t_matrix_setup( actionshist, ml_input, codes_index, sites_index, i_task ):
    
    max_msg = 5
    padding_dim = 200
    actionshist_task = actionshist.iloc[ i_task ]
    task_name = actionshist_task['task_name']
    ml_input_task = ml_input[ml_input['task_name'] == task_name]
    ml_input_task_l = ml_input_task.iloc[0]
    
    # Matrix for actionshist
    good_sites, bad_sites = count_indices( actionshist_task )
    test_matrix_count_std = setup_count_matrix( good_sites, bad_sites, codes_index, sites_index)  
    
    # Matrix for the message
    print ml_input_task
    print ml_input_task['error']
    test_matrix_msg_std = setup_msg_matrix( codes_index, sites_index, ml_input_task_l['error'], ml_input_task_l['site'], 
                                           ml_input_task_l['error_msg_tokenized'], max_msg, padding_dim )
    
    # Matrix for the batch generator
    generator = gen.InputBatchGenerator(ml_input_task, 'label', codes_index, sites_index, padding_dim, batch_size = 1)
    
    for X, y in generator.gen_batches():
        test_matrix_msg_gen = X[0]
        test_matrix_count_gen = X[1].reshape(X[1].shape[1], X[1].shape[2], X[1].shape[3])
    
    # Assert that the arrays are the same
    np.testing.assert_array_equal(test_matrix_count_std, test_matrix_count_gen)
    np.testing.assert_array_equal(test_matrix_msg_std, test_matrix_msg_gen)
    
    return True


def test( input_path='/eos/user/l/llayer/AIErrorLogAnalysis/data/input/', 
         actionshist_path = '/eos/user/l/llayer/AIErrorLogAnalysis/data/actionshist/actionshistory_300719.json', 
         count_test = False, batch_test = False, matrix_setup_test = False, n_matrices = 10):
    
    # Load the input
    mode='tokens'
    input_ml = pd.read_hdf(input_path + 'input_' + mode + '.h5', 'frame')
    sites = pd.read_hdf(input_path + 'input_' + mode + '.h5', 'frame2')
    codes = pd.read_hdf(input_path + 'input_' + mode + '.h5', 'frame3')
    sites_index, codes_index = index.to_index(list(sites['site']), list(codes['error']))
    
    # Load the actionshist
    actionshist = load_data(actionshist_path)
    
    if matrix_setup_test == True:
        
        for i in range(n_matrices):
            rand = random.randint(0, len(input_ml))
            t_matrix_setup(actionshist, input_ml, codes_index, sites_index, rand)
            print( 'Matrix ' + str(i) + ' passed' )
        
        print( 'Pass matrix setup test' )
        
    if count_test == True:
        
        total_counts, total_keys = load_actionshist_counts(actionshist)
        t_count_matrix(input_ml, codes, sites, total_counts, total_keys)
        
    if batch_test == True:

        total_counts, total_keys = load_actionshist_counts(actionshist)
        msg_counts = load_token_counts()
        # Test the standard setup
        padding_size = 200
        generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index, padding_size, batch_size=10)
        t_batch_gen(generator, total_counts, msg_counts)
        print( 'Pass batch generator test in default mode' )
        
        # Test the site summed setup
        generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index, padding_size, batch_size=10, 
                                           mode = 'sum_sites')
        t_batch_gen(generator, total_counts, msg_counts)
        print( 'Pass batch generator test in sum_sites mode' )
        # Test the wf summed setup
        generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index, padding_size, batch_size=10, 
                                           mode = 'sum_sites_errors')
        t_batch_gen(generator, total_counts, msg_counts)
        print( 'Pass batch generator test in sum_sites_errors mode' )

        
        
        
    
    
    
    
