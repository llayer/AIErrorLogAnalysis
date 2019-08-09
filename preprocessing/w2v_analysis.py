import pandas as pd
import numpy as np
import gensim
import itertools
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer
from os import listdir
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


def load_data(unique_msg = True, drop_stageout = True):
    
    time_chunks =  [(20170101, 20171009),(20171011, 20180301),(20180301, 20180601),
                    (20180601, 20181101),(20181101, 20190207),(20190207, 20190801)]
    
    dfs = []
    for time in time_chunks:
        t1, t2 = str(time[0]), str(time[1])
        frame = pd.read_hdf( 'data/messages_filtered_' + t1 + '_' + t2 + '.h5' )
        dfs.append( frame[['task_name', 'error',  'site', 'exit_codes', 'error_msg', 'error_type', 'steps_counter', 'names']])
        
    dfs = pd.concat(dfs, ignore_index=True)
    dfs = dfs.drop_duplicates()
    
    if drop_stageout == True:
        dfs = dfs[dfs['exit_codes'] != 99996] #CHECK!!
    if unique_msg == True:
        dfs = dfs.drop_duplicates(subset=['task_name', 'error', 'site', 'exit_codes', 'error_type', 'steps_counter', 'names'])
        
    return dfs


def tokenize(frame):
    
    # Tokenize the text
    t = TreebankWordTokenizer()
    frame['error_msg_tokenized'] = frame['error_msg'].apply(t.tokenize)
    return frame

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def tokenize_chunks(frame, name, chunks = 1):

    path = 'data/tokenized/'

    if chunks <= 1:
        frame = tokenize(frame)
        frame.to_hdf(path + name + '.h5', 'frame')

    else:
        size_chunk = int(float(len(frame)) / chunks)
        for counter, chunk in enumerate(chunker(frame, size_chunk)):
            print 'Processing chunk ', counter, '/', chunks 
            chunk = tokenize(chunk)
            chunk.to_hdf(path + name + str(counter) + '.h5', 'frame')

def load_tokens(path):
    
    files = [f for f in listdir(path)]
    print files
    tokens = []
    for f in files:
        print f
        frame = pd.read_hdf(path + f, 'frame')
        frame = frame.drop(columns=['error_msg'])
        tokens.append(frame)
    
    #tokens = pd.concat(tokens, ignore_index=True)    
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


def run_word2vec(msgs):
    
    texts_stemmed = list(msgs)
    w2v_sg = gensim.models.Word2Vec(texts_stemmed, size=50, window=5, min_count=5, workers=4, sg=1)
    #w2v_cbow = gensim.models.Word2Vec(texts_stemmed, size=50, window=5, min_count=5, workers=4, sg=0)
    return w2v_sg


def get_avg_vector(frame, model, tfidf = False):
    
    def average_vector(text):
        return np.mean(np.array([model.wv[w] for w in text if w in model]), axis=0)
    
    def average_vector_tfidf(text): 
        return np.mean(np.array([w2vmodel_stemmed.wv[w] * word_weights[w] for w in text if w in w2vmodel_stemmed]), axis=0)
    
    if tfidf == True:
        
        # Get TFIDF
        def dummy(doc):
            return doc
        tfidf = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
        x = tfidf.fit_transform(frame['error_msg_tokenized'])
        word_weights = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
        
        frame['w2v_tfidf'] = frame['error_msg_tokenized'].apply(average_vector_tfidf)
        
    else:
        frame['avg_w2v'] = frame['error_msg_tokenized'].apply(average_vector)
    
    
def tsne(frame):
    
    avg_vec = list(frame)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    tsne_values = tsne_model.fit_transform(list(avg_vec))

    x = []
    y = []
    for value in tsne_values:
        x.append(value[0])
        y.append(value[1])
        
    return x,y


def plot_tsne_error(df, key, n_samples, title = 't-sne'):
    
    sample = df.sample(n_samples)
    
    error_codes = sample[key]
    error_codes_unique = list(sample.groupby(key).size().reset_index(name='size').sort_values(by ='size',
                                                                                              ascending=False  )[key][0:10])
    error_colors = {}
    for counter, code in enumerate(error_codes_unique):
        error_colors[code] = counter
    
    colors = cm.rainbow(np.linspace(0, 1, len(error_codes_unique)))    
    
    x, y = tsne(sample['avg_w2v'])
    
    plt.figure(figsize=(10, 10)) 
    
    legend = []
    for i in range(len(x)):

        error_code = error_codes.iloc[i]
        if error_code not in error_codes_unique:
            continue
        color = error_colors[error_code]
        if error_code not in legend:
            #print error_code, color
            plt.scatter(x[i],y[i], c=colors[color], label = str(error_code))
            legend.append(error_code)
        else:
            plt.scatter(x[i],y[i], c=colors[color])   
    
    plt.legend(loc='upper right')   
    plt.title(title)
        
    plt.show()


def plot_tsne_wf(frame, n_samples, title = 't-sne'):
    
    
    sample = frame.sample(n_samples)
        
    x, y = tsne(sample['avg_w2v'])
    
    plt.figure(figsize=(10, 10)) 
    plt.scatter(x,y)
    plt.title(title)
        
    plt.show()    
    
#def plot_word_cloud():
    
    
    
    
    
    
