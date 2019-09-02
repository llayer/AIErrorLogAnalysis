import pandas as pd
from collections import Counter
from os import listdir
import itertools


def load_tokens(path):
    
    files = [f for f in listdir(path)]
    print files
    tokens = []
    for f in files:
        print f
        frame = pd.read_hdf(path + f, 'frame')
        tokens.append(frame)
    return tokens


def get_frequencies(df, col_name):
    # Count the word occurences
    str_frequencies = pd.DataFrame(list(Counter(filter(None,list(itertools.chain(*df[col_name]
                                                                            )))).items()),columns=['word','count'])
    return str_frequencies.sort_values(by ='count',ascending=False  )


def get_counts(df, col_name):
    
    # Count the word occurences
    str_frequencies = Counter(filter(None,list(itertools.chain(*df[col_name]))))
    return str_frequencies  


def get_low_freq_words(df, col_name, minimum_count):
    
    str_frequencies = get_frequencies(df,col_name)
    return set(str_frequencies[str_frequencies['count'] < minimum_count]['word'])


def check_first_and_last_char(word):
    if word[0] in "':/.-=": 
        #print word
        word = word.replace(word[0], '')
        #print word
    if len(word) > 1:
        if word[-1] in "':/.-": 
            word = word.replace(word[-1], '')
    # lowercase
    #word = word.lower()
    return word


# Remove punctuation and kick out words that occur less than a certain threshold
def clean_data(tokens, low_frequency_words):
    words = filter(lambda word: word not in '``#\'"\'\'|==,|--$;:=+><[!@]|&?{}...%(.)""()==========' 
                   and word not in low_frequency_words, tokens)
    words_cleaned = [check_first_and_last_char(word) for word in words]
    #words_cleaned = filter(lambda word: word.isalpha() == False, words_cleaned)
    words_cleaned = filter(lambda word: word not in 'abcdefghijklmnopqrstuvwxyzn=', words_cleaned)
    return words_cleaned


def clean( df, col_name, minimum_count ):

    print( 'Extracting low frequency words' )
    low_freq_words = get_low_freq_words(df, col_name, minimum_count)
    print( 'Cleaning tokens' )
    df[col_name] = df[col_name].apply(clean_data, args=[low_freq_words])
    print( 'Getting vocabulary' )
    vocab = get_frequencies(df, col_name)
    return vocab


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def store_filtered_tokens(frame, path, name, vocab, chunks = 1):
   
    
    vocab.to_hdf(path + 'vocab.h5', 'frame')
    
    if chunks <= 1:
        frame = tokenize(frame)
        frame.to_hdf(path + name + '.h5', 'frame', mode = 'w')

    else:
        size_chunk = int(float(len(frame)) / chunks)
        for counter, chunk in enumerate(chunker(frame, size_chunk)):
            print 'Processing chunk ', counter, '/', chunks 
            chunk.to_hdf(path + name + str(counter) + '.h5', 'frame', mode = 'w')  