import pandas as pd
from os import listdir


def load_tokens(path):
    
    files = [f for f in listdir(path)]
    print files
    tokens = []
    for f in files:
        if 'vocab' in f:
            continue
        print f
        frame = pd.read_hdf(path + f, 'frame')
        tokens.append(frame)
    return tokens


def rules(sequence):
    
    fatal = 'Fatal Exception'
    perf = 'PerformanceKill'
    
    if perf in seq and fatal in seq:
        return 0
    elif perf in seq:
        return 1
    elif fatal in seq:
        return 2
    else:
        return 3

    
def unique_message(row):
    
    error = row['error']
    error_type_string = row['error_type_string']
    exit_codes = row['exit_codes']
    error_msg = row['error_msg']
    error_type = row['error_type']
    names = row['names']
    performances = row['performance']
    
    fatal = 'Fatal Exception'
    perf = 'PerformanceKill'
    
    fatal_msg = []
    perf_msg = []
    errors = []
    
    performance = performances[0]
    for p, n in zip(performances, names):
        if n == 'cmsRun1':
            performance = p
            break
    
    for i in range(len(exit_codes)):
        
        if perf in error_type[i]:
            perf_msg.append(i)
        if fatal in error_type[i]:
            fatal_msg.append(i)
        if error == exit_codes[i]:
            errors.append(i)
    
    # 1. Rule both performance kill and fatal exception
    if len(fatal_msg) > 0 and len(perf_msg) > 0:
        msg = error_msg[perf_msg[0]] + error_msg[fatal_msg[0]]
        exit_code = exit_codes[perf_msg[0]]
        err_type = error_type[perf_msg[0]] + error_type[fatal_msg[0]]
    # 2. Rule at least one fatal exception
    elif len(fatal_msg) > 0:
        msg = error_msg[fatal_msg[0]]
        exit_code = exit_codes[fatal_msg[0]]
        err_type = error_type[fatal_msg[0]]
    # 3. Rule at least one performance kill
    elif len(perf_msg) > 0:
        msg = error_msg[perf_msg[0]]
        exit_code = exit_codes[perf_msg[0]]
        err_type = error_type[perf_msg[0]]
    # 4. Else take the message with the same error code
    else:
        msg = error_msg[errors[0]]
        exit_code = exit_codes[errors[0]]
        err_type = error_type[errors[0]]
        
    x = {}
    x['exit_code'] = exit_code
    x['error_msg'] = msg
    x['error_type'] = err_type
    x['performance'] = performance
    
    return x
    

def clean_error_type(error_type):
    
    if len(error_type) > 50:
        return str('TypeError').decode('utf-8')
    else:
        return error_type
    

    
def select_message(tokens):
    
    # Clean up the tokens
    print( 'Setting up performance vector' )
    tokens = tokens.drop(['NumberOfThreads', 'NumberOfStreams', 'readTotalSecs', '_c0'], axis = 1)
    performance = [ 'peakvaluerss', 'peakvaluevsize', 'writeTotalMB', 'readPercentageOps', 'readAveragekB', 'readTotalMB',
                   'readNumOps', 'readCachePercentageOps', 'readMBSec', 'writeTotalSecs', 'readMaxMSec',
                   'TotalJobCPU', 'TotalInitCPU', 'TotalEventCPU', 'AvgEventCPU', 'EventThroughput',
                   'TotalInitTime', 'AvgEventTime', 'MinEventCPU', 'MaxEventTime', 'TotalJobTime',
                   'TotalLoopCPU', 'MinEventTime', 'MaxEventCPU']
    tokens['performance'] = tokens[performance].apply(tuple, axis=1) 
    tokens['error_type'] = tokens['error_type'].apply(clean_error_type)
    
    print( 'Aggregating' )
    tokens = tokens.groupby(['task_name', 'error', 'site'], as_index=False)['exit_codes', 'error_msg','error_type',
                                                                            'steps_counter', 'names', 
                                                                            'performance'].agg(lambda x: list(x))
    
    print( 'Select unique message' )
    tokens['error_type_string'] = [','.join(map(str, l)) for l in tokens['error_type']]
    unique_msg = tokens.apply(unique_message, axis=1).to_frame('unique_msg')
    unique_msg = unique_msg['unique_msg'].apply(pd.Series)
    #tokens = tokens.drop(['performance'], axis=1)
    tokens = tokens.drop(['exit_codes', 'steps_counter', 'names', 'error_msg', 'error_type', 'performance', 
                          'error_type_string'], axis=1)
    tokens = tokens.join(unique_msg)

    return tokens
    
    
    
    