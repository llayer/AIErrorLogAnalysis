import pandas as pd

def load_data(path):
    
    # Load the actionshistory
    data = pd.read_json(path, orient='index')

    # Reset index
    data_index_reset = data.reset_index()
    data_index_reset = data_index_reset.rename(columns={'index': 'task_name'})
    
    return data_index_reset


def loop_over_sites(sites, task_name, ignore_neg_code, site_state, label):
    key = []
    for exit_code, site_dict in zip(sites.keys(), sites.values()):
        if ignore_neg_code == True:
            if int(exit_code) == -1:
                continue
        for site, count in site_dict.iteritems():
            key.append((task_name, exit_code, site, site_state, count, label))
    return key


def map_to_key(row, ignore_neg_code):
    
    task_name = row['task_name']
    errors = row['errors']
    good_sites = errors['good_sites']
    bad_sites = errors['bad_sites']
    try:
        label = row['action_binary_encoded']
    except:
        label = 'NaN'
    
    good_sites_key = loop_over_sites(good_sites, task_name, ignore_neg_code, 'good', label)
    bad_sites_key = loop_over_sites(bad_sites, task_name, ignore_neg_code, 'bad', label)
        
    return list(set(good_sites_key + bad_sites_key))


def expand_key(key):
    return key[0], key[1], key[2], key[3], key[4], key[5]


def get_keys(data, ignore_neg_code = True):
    
    data['keys'] = data.apply(lambda x: map_to_key(x, ignore_neg_code = ignore_neg_code), axis = 1)
    keys = data['keys'].apply(pd.Series).unstack().reset_index(drop=True).dropna()
    keys = keys.to_frame('key')
    keys['task_name'], keys['error'], keys['site'], keys['site_state'], \
    keys['count'], keys['label'] = zip(*keys['key'].map(expand_key))
    keys = keys.drop(columns=['key'])
    keys.site = keys.site.str.encode('utf-8')
    
    return keys


