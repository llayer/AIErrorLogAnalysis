import numpy as np
import InputGenerator
import InputBatchGenerator


def pruned_to_index(dict_old, list_new):

    dict_pruned = { k : len(errors_pruned) for k,v in dict_old.items() if v not in list_new }
    list_new = [ k for k, v in dict_old.items() if v in list_new ]
    dict_new = {k: v for v, k in enumerate(list_new)}
    
    print( len(dict_new), len(dict_old), len(dict_pruned))
    
    assert( (len(dict_new) + len(dict_pruned)) == len(dict_old) )
    
    return { **dict_new, **dict_pruned }



def prune_counts(mat, ax, threshold, unweighted = True):
    
    
    if unweighted == True:
        m_zero = mat > 0
        m_binary = m_zero * 1
        summed = m_binary.sum(axis=ax).sum(axis=0)
        mask_larger = summed > threshold
    
    else:
        summed = mat.sum(axis=ax).sum(axis=0)
        mask_larger = summed > threshold
    
    """
    if ax == 1:
        mat_larger = mat[:, : , mask_larger]
    else:
        mat_larger = mat[:, mask_larger, :]
    """                 
    p = np.where(mask_larger == True )
    p = list(p[0])
    """
    p_str = []
    for error in p:
        p_str.append( str(error).encode('utf-8') )
    """
    return p


def prune_neg(mat):
    
    print( mat.sum() )
    
    summed = mat.sum(axis=0)
    neg = gen.codes['-1'.encode('utf-8')]
    neg_codes = np.delete(summed, (neg), axis=0)
    non_zeros = neg_codes.sum(axis = 0) > 0
    
    """
    zeros = neg_codes.sum(axis = 0) == 0
    zeros_sum = zeros.sum()
    
    pruned_mat = mat[:, :, non_zeros]
    pruned_zeros_mat = mat[:, :, zeros].sum(axis = 2).reshape(mat.shape[0], mat.shape[1], 1)

    
    return np.concatenate([pruned_mat, pruned_zeros_mat], axis = 2)
    """
    
    p = np.where(non_zeros == True )
    l = list(p[0])
    l.append(-1)
    """
    p_str = []
    for error in l:
        p_str.append( str(error).encode('utf-8') )
    """
    
    return p



def prune(mat, errors, site, error_threshold = 0, site_threshold = 0, neg = False, unweighted = True ):
    
    
    
    counts = mat.sum()
    
    if neg == True:
        non_neg = prune_neg(mat)
    
    print ('Start site')
    if site_threshold > 0:
        site_pruned = prune_site_error(mat, 1, site_threshold, unweighted)
        #assert( counts == mat.sum())
    
    print ('End site')
    
    if error_threshold > 0:
        errors_pruned = prune_site_error(mat, 2, error_threshold, unweighted)

    print ('End error')
    
    print(len(non_neg), len(site_pruned), len(errors_pruned))

    
    errors = pruned_to_index(errors, errors_pruned)
    site_combined = list(set(non_neg + site_pruned))
    sites = pruned_to_index(site, site_combined)
        
    return errors, sites


def get_max_msg_pruned(mat):
    return np.max(mat.sum(axis=1)), np.max(mat.sum(axis=2).sum(axis=1))

