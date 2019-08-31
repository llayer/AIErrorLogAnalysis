import pandas as pd
#import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer


def load_data(path, unique_msg = True):
    
    time_chunks =  [(20170101, 20171009),(20171011, 20180301),(20180301, 20180601),
                    (20180601, 20181101),(20181101, 20190207),(20190207, 20190801)]
    
    dfs = []
    for time in time_chunks:
        print( time )
        t1, t2 = str(time[0]), str(time[1])
        frame = pd.read_hdf( path + 'messages_filtered_' + 
                            t1 + '_' + t2 + '.h5' )
        dfs.append( frame )
        
    dfs = pd.concat(dfs, ignore_index=True)
    dfs = dfs.drop_duplicates()

    if unique_msg == True:
        dfs = dfs.drop_duplicates(subset=['task_name', 'error', 'site', 'exit_codes', 'error_type', 'steps_counter', 'names'])
        
    return dfs


def tokenize(frame):
    
    # Tokenize the text
    t = TreebankWordTokenizer()
    frame['error_msg'] = frame['error_msg'].apply(t.tokenize)
    return frame


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def tokenize_chunks(frame, path, name, chunks = 1):

    if chunks <= 1:
        frame = tokenize(frame)
        frame.to_hdf(path + name + '.h5', 'frame', mode = 'w')

    else:
        size_chunk = int(float(len(frame)) / chunks)
        for counter, chunk in enumerate(chunker(frame, size_chunk)):
            print 'Processing chunk ', counter, '/', chunks 
            chunk = tokenize(chunk)
            chunk.to_hdf(path + name + str(counter) + '.h5', 'frame', mode = 'w')
            
            