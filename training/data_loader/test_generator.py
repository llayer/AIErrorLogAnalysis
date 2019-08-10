import numpy as np
import input_batch_generator as gen
import index 



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
    
    
