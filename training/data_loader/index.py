import numpy as np


def to_index(sites, codes):

    sites_index = {k: v for v, k in enumerate(sites)}
    codes_index = {k: v for v, k in enumerate(codes)}
    return sites_index, codes_index


def tiers_to_index(sites):
    tiers_to_index = {'T0' : 0, 'T1' : 1, 'T2' : 2, 'T3' : 3}
    sites_tiers = {}
    for site in sites:
        tier = site[0:2].decode('utf8')
        if tier in tiers_to_index:
            sites_tiers[site] = tiers_to_index[tier]
        else:
            sites_tiers[site] = 4

    return sites_tiers


def prune_to_index(codes, sites, only_unknown = False, counts = False, error_threshold = 0, site_threshold = 0):
    
    all_sites = list(sites['site'])
    all_codes = list(codes['error'])
    good_sites = list(sites['site'])
    good_codes = list(codes['error'])
    
    if only_unknown == True:
        informative_sites = list(sites[sites['only_unknown'] == False]['site'])
        good_sites = list(set(informative_sites) & set(good_sites))  

    if site_threshold > 0:
        if counts == False:
            frequent_sites = list(sites[sites['frequency'] > site_threshold]['site'])
        else:
            frequent_sites = list(sites[sites['counts'] > site_threshold]['site'])
        good_sites = list(set(frequent_sites) & set(good_sites))  
            
    if error_threshold > 0:
        if counts == False:
            frequent_errors = list(codes[codes['frequency'] > error_threshold]['error'])
        else:
            frequent_errors = list(codes[codes['counts'] > error_threshold]['error'])    
        good_codes = list(set(frequent_errors) & set(good_codes)) 
        
    # Get the pruned sites and codes
    pruned_sites = list(set(all_sites) - set(good_sites))
    pruned_codes = list(set(all_codes) - set(good_codes))
    
    # Index the results
    good_sites_index = {k: v for v, k in enumerate(good_sites)}
    pruned_sites_index = {k: len(good_sites) for k in pruned_sites}
    good_codes_index = {k: v for v, k in enumerate(good_codes)}
    pruned_codes_index = {k: len(good_codes) for k in pruned_codes}    
        
    def merge_dicts(x, y):
        z = x.copy()   
        z.update(y) 
        return z
    
    codes_index = merge_dicts(good_codes_index, pruned_codes_index)
    sites_index = merge_dicts(good_sites_index, pruned_sites_index)
    
    return sites_index, codes_index


"""
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
"""



