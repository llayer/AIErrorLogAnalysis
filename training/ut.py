import pandas as pd
import json
import itertools
import math
import numpy as np



###########################
# Sites and error codes
###########################

def get_sites(frame):
    
    def list_of_sites(row, att):
        sites = row[att].values()
        sites_list = []
        if len(sites) != 0:
            sites_list = [item.keys() for item in sites]
        else:
            sites_list = ['NA']
        return sites

    good_site_sites = frame['errors'].apply(lambda x:  list_of_sites(x, 'good_sites')).tolist()
    bad_site_sites = frame['errors'].apply(lambda x:  list_of_sites(x, 'bad_sites')).tolist()
  
    good_site_sites = list(itertools.chain.from_iterable(good_site_sites))    
    bad_site_sites = list(itertools.chain.from_iterable(bad_site_sites))    
    
    good_site_sites = sorted(set(list(itertools.chain.from_iterable((good_site_sites)))))
    bad_site_sites = sorted(set(list(itertools.chain.from_iterable(bad_site_sites))))
    
    good_site_sites = [x.encode('utf-8') for x in good_site_sites]
    bad_site_sites = [x.encode('utf-8') for x in bad_site_sites]
   
    return good_site_sites, bad_site_sites


def get_exit_codes(frame):
	
    def exit_codes(row, att):
        sites = row[att]
        return [ site for site in sites ] if len(sites.keys()) != 0 else ['0']


    good_site_codes = frame['errors'].apply(lambda x:  exit_codes(x, 'good_sites')).tolist()
    bad_site_codes = frame['errors'].apply(lambda x:  exit_codes(x, 'bad_sites')).tolist()
    
    good_site_codes = sorted(set(list(itertools.chain.from_iterable(good_site_codes))),key=int)
    bad_site_codes = sorted(set(list(itertools.chain.from_iterable(bad_site_codes))),key=int)
    
    good_site_codes = [x.encode('utf-8') for x in good_site_codes]
    bad_site_codes = [x.encode('utf-8') for x in bad_site_codes]

    return good_site_codes, bad_site_codes


###########################
# Labels
##########################

def xrootd_fnc(x, column):
    # if isinstance(x.keys(), dict): 
    if column in x.keys():
        return str(x[column])

    else:
        return str('NaN')

def xrootd_encoded(xrootd):
    
    encoder = -99
    if ('enabled' in xrootd):
        encoder = 0
    elif ('disabled') in xrootd:
        encoder = 1
    else:
        encoder = 2
    return encoder
    
def splitting_fnc(x, column):
    if column in x.keys():
        return str(x[column])

    else:
        return '1x'

def splitting_encoded(split):
    
    encoder = -99
    if ('1x' in split):
        encoder = 0
    elif ('2x' in split) or ('3x' in split):
        encoder = 1
    else:
        encoder = 2       
    return encoder
    
def memory_fnc(x, column):
    # if isinstance(x.keys(), dict): 
    if column in x.keys():
        return str(x[column])
    else:
        return str('NaN')

    
def memory_encoded(memory):
    
    encoder = -99
    if (memory == 'NaN') or (memory == ''):
        encoder = 3
        return encoder

    memory_int = float(memory)
    if ( memory_int < 9999):
        encoder = 0
    elif ( memory_int > 9999 and memory_int < 16001):
        encoder = 1
    else:
        encoder = 2       
    return encoder    
    
    
def action_encoded(action):
    
    encoder = -99
    if 'acdc' in action:
        encoder = 0
    elif 'clone' in action:
        encoder = 1
    else:
        encoder = 2
    return encoder
    
def merge_labels(x, features):
    merged_label = '_'.join(x[features])

    return merged_label

def set_labels(frame):

    # Add column with splitting categorical levels
    frame['splitting'] = frame['parameters'].apply(lambda x: splitting_fnc(x, 'splitting'))
    # Encode splitting
    frame['splitting_encoded'] = frame['splitting'].apply(lambda x: splitting_encoded(x)) 
    # Add column with xrootd categorical levels
    frame['xrootd'] = frame['parameters'].apply(lambda x: xrootd_fnc(x, 'xrootd'))
    # Add xrootd encoded
    frame['xrootd_encoded'] = frame['xrootd'].apply(lambda x: xrootd_encoded(x)) 
    # Add column with memory categorical levels
    frame['memory'] = frame['parameters'].apply(lambda x: memory_fnc(x, 'memory'))
    # Add memory encoded
    frame['memory_encoded'] = frame['memory'].apply(lambda x: memory_encoded(x)) 
    # Set 'action' as the target
    frame['action'] = frame['parameters'].apply(lambda x: x['action'])
    # Add memory encoded
    frame['action_encoded'] = frame['action'].apply(lambda x: action_encoded(x))
    # Target categorical levels
    frame['action_split'] = frame.apply(lambda x: merge_labels(x, ['action','splitting']), axis=1)
    # Encode target categorical levels
    target_categories = sorted(list(set(frame['action_split'])))
    frame['action_split_encoded'] = \
    frame['action_split'].astype(pd.api.types.CategoricalDtype(categories =target_categories)).cat.codes
    # create a binary classification column
    frame['action_binary'] = frame['action'].apply(lambda x: 'acdc' if x == 'acdc' else 'non_acdc')
    # create a binary classification column
    frame['action_binary_encoded'] = frame['action'].apply(lambda x: 0 if x == 'acdc' else 1)
    
def set_binary_labels(frame):
    # Set 'action' as the target
    frame['action'] = frame['parameters'].apply(lambda x: x['action'])    
    # create a binary classification column
    frame['action_binary_encoded'] = frame['action'].apply(lambda x: 0 if x == 'acdc' else 1)





