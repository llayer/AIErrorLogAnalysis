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



def load_actionshist_counts(path_actionshist =
                            '/eos/user/l/llayer/AIErrorLogAnalysis/data/actionshist/actionshistory_300719.json'):
    
    # Load the actionshistory
    print( 'Loading actionshist and generating keys' )
    actionshist = load_data(path_actionshist)
    set_binary_labels(actionshist)
    actionshist_keys = get_keys(actionshist, ignore_neg_code = False)
    total_counts = actionshist_keys['count'].sum()
    total_keys = len(actionshist_keys['count'])
    
    del actionshist_keys
    return total_counts, total_keys


def count_matrix_sum(gen, total_counts, total_keys):
        
    X, _ = gen.get_counts_matrix()
    binary = X > 0
    
    assert(total_counts == X.sum())
    assert(total_keys == binary.sum())
    
    del X
    del binary
    
    return True


def t_count_matrix(input_ml, sites, codes):
    
    total_counts, total_keys = load_actionshist_counts()
    
    # Standard count matrix
    padding_size = 200
    sites_index, codes_index = index.to_index(list(sites['site']), list(codes['error']))
    generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index,
                                                      padding_size, batch_size=10)
    count_matrix_sum(generator, total_counts, total_keys)

    # Prune without counts
    sites_index, codes_index = index.prune_to_index(codes, sites, only_unknown = True, counts = False, error_threshold = 100, site_threshold = 1000)
    generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index,
                                                      padding_size, batch_size=10)    
    count_matrix_sum(generator, total_counts, total_keys)
    
    
    # Prune with counts
    sites_index, codes_index = index.prune_to_index(codes, sites, only_unknown = True, counts = True, error_threshold = 1000, site_threshold = 10000)
    generator = gen.InputBatchGenerator(input_ml, 'label', codes_index, sites_index,
                                                      padding_size, batch_size=10) 
    count_matrix_sum(generator, total_counts, total_keys)

    
    

def t_batch_gen_sum(gen):
    
    sum_counts = []
    sum_msg = []
    
    for X,y in gen.gen_batches():
        
        n_msgs = X[0].sum(axis=4).sum(axis=3) > 0
        sum_msg.append(n_msgs.sum())
        sum_counts.append(X[1].sum())
        
    print np.array(sum_counts).sum()
    print np.array(sum_msg).sum()




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


def to_num( good, bad, codes, sites ):
    
    
    matrix = np.zeros((len(codes), len(sites), 2))
    
    for entry in good:
        matrix[ codes[entry[0]], sites[entry[1]], 0 ] = entry[2]
    for entry in bad:
        matrix[ codes[entry[0]], sites[entry[1]], 1 ] = entry[2]    
        
    return matrix


def test_count_matrix_setup( actionshist, ml_input, codes, sites, i_task ):
    
    actionshist_task = actionshist.iloc[ i_task ]
    task_name = actionshist_task['task_name']
    ml_input_task = ml_input[ml_input['task_name' == task_name]]
    
    # Matrix for actionshist
    indices = count_indices( actionshist_task )
    test_matrix = to_num( indices[0], indices[1], codes, sites)  
    
    
    # Matrix for the batch generator
    generator = gen.InputBatchGenerator(ml_input_task, 'label', codes_index, sites_index, 200, batch_size = 1)
    matrix = generator.get_counts_matrix()
    
    # Assert that the arrays are the same
    if np.testing.assert_array_equal(test_matrix, test_matrix_gen) == True:
        print( 'Arrays are the same' )

"""

def test(path_actionshist, path_input):
    

    
    # Load the ml input 
    ml_input = pd.read_hdf(path_input, 'frame')
    sites = pd.read_hdf(path_input, 'frame2')
    codes = pd.read_hdf(path_input, 'frame3')
    
"""
    
    


def test_count_matrix( gen, i_task ):

    # Generate the matrix for a single task
    codes = gen.codes
    sites = gen.sites
    test_task = gen.inspect_single_task( i_task )
    indices = count_indices( test_task )
    test_matrix = to_num( indices[0], indices[1], codes, sites)

    # Automatic generation
    test_gen = InputBatchGenerator.InputBatchGenerator(gen.actionshistory, 'action_binary_encoded', \
                                                    gen.codes, gen.sites, gen.dim_msg)
    
    full_matrix_gen = test_gen.count_matrix()[0]
    test_matrix_gen = full_matrix_gen[ i_task ]
    
    # Assert that the arrays are the same
    if np.testing.assert_array_equal(test_matrix, test_matrix_gen) == True:
        print( 'Arrays are the same' )
    
    # Check the sites and indices:
    for index in indices[0]:
        assert( test_matrix_gen[codes[index[0]], sites[index[1]], 0] == index[2] )
    for index in indices[1]:
        assert( test_matrix_gen[codes[index[0]], sites[index[1]], 1] == index[2] )  
        
    # Assert that the number of filled entries in the matrix are correct
    codes, sites, keys = gen.prune_error_sites()
    assert( np.count_nonzero(full_matrix_gen) == keys )
    
    # Assert that the sites2tiers works properly
    sites_T = gen.sites_to_tiers(gen.sites)
    test_tier_gen = InputBatchGenerator.InputBatchGenerator(gen.actionshistory, 'action_binary_encoded', gen.codes, sites_T, gen.dim_msg)
    matrix_T = test_tier_gen.count_matrix()
    
    assert( full_matrix_gen.sum() == matrix_T.sum() )
    
    
