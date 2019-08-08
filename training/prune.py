import numpy as np
import InputGenerator
import InputBatchGenerator


def pruned_to_index(dict_old, list_new):

    # Map the old dictionary to a new one
   
    # Get a dictionary with the pruned sites and the new index
    dict_pruned = { k : len(list_new) for k,v in dict_old.items() if v not in list_new }
    # Get the keys of the old dictionary if they are in the new list
    list_new = [ k for k, v in dict_old.items() if v in list_new ]
    # Make a new enumeration for the non pruned keys
    dict_new = {k: v for v, k in enumerate(list_new)}
        
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
    
    p = np.where(mask_larger == True )
    l = list(set(p[0]))

    return l


def prune_neg(mat, codes):
    
    # Sum over tasks
    summed = mat.sum(axis=0)
    # Delete the -1 codes from the matrix
    neg = codes['-1'.encode('utf-8')]
    neg_codes = np.delete(summed, (neg), axis=0)
    # Get the sites that are non zero without -1 errors
    non_zeros = neg_codes.sum(axis = 0) > 0
    p = np.where(non_zeros == True)

    # Return a list with non-zero sites
    l = list(p[0])
    
    return l



def prune(mat, errors, site, error_threshold = 0, site_threshold = 0, neg = False, unweighted = True ):
    
    counts = mat.sum()
    site_pruned = []
    errors_pruned = []
    non_neg = []
    if neg == True:
        non_neg = prune_neg(mat, errors)
        print( non_neg )
    print ('Start site')
    if site_threshold > 0:
        site_pruned = prune_counts(mat, 1, site_threshold, unweighted)
        #assert( counts == mat.sum())
    
    print ('End site')
    
    if error_threshold > 0:
        errors_pruned = prune_counts(mat, 2, error_threshold, unweighted)
        print( errors_pruned )

    print ('End error')
    
    #print(len(non_neg), len(site_pruned), len(errors_pruned))

    
    errors = pruned_to_index(errors, errors_pruned)
    site_combined = list(set(site_pruned) & set(non_neg)) 
    sites = pruned_to_index(site, site_combined)
        
    return sites, errors



