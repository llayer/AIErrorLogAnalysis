import pandas as pd
import numpy as np

do_tokenization = False
do_cleaning = False
do_selection = False
do_embeddings = False
do_indexing = False
do_input = True


# Defines the input experiments for the machine learning
EXPERIMENTS = [
    {'NAME': 'NOMINAL', 'DIM':50, 'VOCAB': -1, 'ALGO': 'sg'} ,
    #{'NAME': 'VAR_DIM', 'DIM':32, 'VOCAB': -1, 'ALGO': 'sg'} ,
    #{'NAME': 'VAR_VOCAB', 'DIM':50, 'VOCAB': 10000, 'ALGO': 'sg'}
]

# PATHS to the files
PATH_ACTIONSHIST = '/eos/user/l/llayer/AIErrorLogAnalysis/data/actionshist/actionshistory_300719.json'
PATH_MESSAGES = '/eos/user/l/llayer/AIErrorLogAnalysis/data/filtered_messages/'
PATH_TOKENS = '/eos/user/l/llayer/AIErrorLogAnalysis/data/tokenized_tmp/'
PATH_CLEANED_TOKENS = '/eos/user/l/llayer/AIErrorLogAnalysis/data/prefilter/'
PATH_SELECTED = '/eos/user/l/llayer/AIErrorLogAnalysis/data/selected/'
PATH_MODELS = '/eos/user/l/llayer/AIErrorLogAnalysis/data/word2vec/models/'
PATH_ENCODING = '/eos/user/l/llayer/AIErrorLogAnalysis/data/word2vec/'
PATH_INPUT = '/eos/user/l/llayer/AIErrorLogAnalysis/data/input/'


if do_tokenization == True:
    
    print( 'Start tokenization' )
    import tokens
    data = tokens.load_data(PATH_MESSAGES)
    tokens.tokenize_chunks(data, PATH_TOKENS, 'tokens', 10)


if do_cleaning == True:
    
    print( 'Start cleaning' )
    import clean
    tokens = clean.load_tokens(PATH_TOKENS)
    tokens = pd.concat(tokens)
    vocab = clean.clean(tokens, 'error_msg', 10)
    clean.store_filtered_tokens(tokens, PATH_CLEANED_TOKENS, 'tokens', vocab, 10)

    
if do_selection == True:
    
    print( 'Start selection' )
    import message_selection
    tokens = message_selection.load_tokens(PATH_CLEANED_TOKENS)
    tokens = pd.concat(tokens)
    tokens = message_selection.select_message(tokens)
    tokens.to_hdf(PATH_SELECTED + 'selected.h5', 'frame', mode = 'w')
    
    
if do_embeddings == True:
    
    print( 'Start embeddings' )
    import word2vec
    tokens = word2vec.load_tokens(PATH_CLEANED_TOKENS)
    tokens = pd.concat(tokens)
    
    dims = []
    algos = []
    for exp in EXPERIMENTS:
        dims.append(exp['DIM'])
        algos.append(exp['ALGO'])
        
    for dim in list(set(dims)):
        for algo in list(set(algos)):
            print( 'Creating embeddings for:', algo, dim )
            model_path = PATH_MODELS + 'model_' + algo + '_' + str(dim) + '.model'
            word2index = word2vec.run_word2vec(tokens, embedding_size = dim, algo = algo, model_path = model_path)
        

if do_indexing == True:
    
    print( 'Start indexing' )
    import word2vec
    
    for exp in EXPERIMENTS:
        print( 'Indexing:', exp )
        model_path = PATH_MODELS + 'model_' + exp['ALGO'] + '_' + str(exp['DIM']) + '.model'
        model = word2vec.load_model(model_path)
        tokens = pd.read_hdf(PATH_SELECTED + 'selected.h5')
        name = exp['NAME']
        word2vec.encode_tokens(tokens, model, exp['DIM'], max_words = exp['VOCAB'], 
                               name = name, avg_vec = True, store=True, path=PATH_ENCODING)
        
            
if do_input == True:
    
    print( 'Start input generation' )
    import input_generator
    for exp in EXPERIMENTS:
        print( 'Input:', exp )    
        path_tokens = PATH_ENCODING + 'tokens_index_' + exp['NAME'] + '.h5'
        input_generator.create_input(PATH_ACTIONSHIST, path_tokens, name = exp['NAME'], store_path = PATH_INPUT)
    
    
    
    
    
    