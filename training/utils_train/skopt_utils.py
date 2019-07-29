'''
    skopt optimization utility functions
'''

import numpy as np
import json

def get_best_params(search_result,names):
    '''
    @param search_result - result from skopt optimization
    @param names - list of all dimensions (order matters!)
    return dict {'param1':int, 'param2':int, ...}
    '''
    outputDict = {}
    for i,nr in enumerate(search_result.x):
        outputDict[ names[i] ] = str(nr)
    return outputDict

def create_skopt_results_string(search_result, prior_names, num_calls, savepath=None):
    '''
    @param search_result - result from skopt optimization
    @param prior_names - list of all dimensions (order matters!)
    @param num_calls - number of calls used in skopt optimization
    @param savepath - path where to save this string
    return string
    '''
    s = ''
    s += '  ::: ALL PARAMETERS :::'
    s += '{:3}'.format('')
    indexes = np.arange(num_calls)
    sorted_results = sorted(zip(search_result.func_vals, indexes, search_result.x_iters))
    for name in prior_names:
        s += '{} '.format(name)
    s += '\n'
    for fitness_value,index,parameter_values in sorted_results:
        s += 'result value: {:3.3} on step:{:3}. {:3} params: '.format(fitness_value,index,'')
        for x in parameter_values:
            x = str(x)
            if len(x) < 10:
                s += '{:10.5} '.format(x)
            else:
                s += '{:11.8}'.format(x)
        s += '\n'
    s += '  ::: BEST SCORE :::\n'
    s += str(search_result.fun) + '\n'
    s += '  ::: BEST PARAM :::\n'
    best_param = get_best_params(search_result,prior_names)
    s += json.dumps(best_param,indent=4)
    if savepath is not None:
        with open(savepath,'w') as f:
            f.write(s)
    return s