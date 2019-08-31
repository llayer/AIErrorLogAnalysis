import pandas as pd
import numpy as np
import gensim
from collections import Counter
import itertools
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE  
    

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


def get_tfidf(frame):
    
    # Get TFIDF
    def dummy(doc):
        return doc
    tfidf = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    x = tfidf.fit_transform(frame['error_msg'])
    word_weights = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    return word_weights


def get_vocab(df):
    
    # tfidf 
    tfidf_weights = get_tfidf(df)
    
    # Count the word occurences
    str_frequencies = pd.DataFrame(list(Counter(filter(None,list(itertools.chain(*df['error_msg']
                                                                            )))).items()),columns=['word','count'])
    
    str_frequencies['tfidf_score'] = str_frequencies['word'].apply(lambda x: tfidf_weights[x])
    
    return str_frequencies.sort_values(by ='count',ascending=False  )


def reduce_vocab(df, minimum_count = -1, max_words = -1):
    
    vocab = get_vocab(df)
    
    if minimum_count > 0:
        vocab = set(vocab[vocab['count'] > minimum_count]['word'])

    if max_words > 0:
        if len(vocab) > max_words:
            return vocab[0:max_words]
    
    return vocab
    
    
def get_avg_vector(frame, model, tfidf = False):
    
    def average_vector(text):
        return np.mean(np.array([model.wv[w] for w in text if w in model]), axis=0)
    
    def average_vector_tfidf(text): 
        return np.mean(np.array([w2vmodel_stemmed.wv[w] * word_weights[w] for w in text if w in w2vmodel_stemmed]), axis=0)
    
    if tfidf == True:
        
        frame['avg_tfidf'] = frame['error_msg'].apply(average_vector_tfidf)
        
    else:
        frame['avg'] = frame['error_msg'].apply(average_vector)

    
def run_word2vec(tokens, embedding_size = 50, minimum_count = -1, max_words = -1, algo = 'sg', model_path = 'test.model'):
    
    if algo == 'sg':
        w2v_algo = 1
    else:
        w2v_algo = 0
    
    # Reduce vocabulary
    if minimum_count > 0 or max_words > 0:
        print( "Reduce the vocab" )
        reduced_vocab = reduce_vocab(tokens, minimum_count = minimum_count, max_words = max_words)
        def clean_data(tokens, vocab):
            words_cleaned = filter(lambda word: word in vocab, tokens)
            return words_cleaned
        tokens['error_msg'] = tokens['error_msg'].apply(clean_data, args=[reduced_vocab])
    
    print( "Run word2vec" )
    # Run the word2vec model
    texts_stemmed = list(tokens['error_msg'])
    # 1 for skip gram, 0 for bow
    model = gensim.models.Word2Vec(texts_stemmed, size=embedding_size, window=5, min_count=5, workers=4, sg=w2v_algo)
    model.save(model_path)
    
    return model
    
    
def load_model(path):
    model = gensim.models.Word2Vec.load(path)
    return model


def encode_tokens(tokens, model, embedding_dim, name = 'test', minimum_count = -1, max_words = -1, 
                  avg_vec=False, store = False, path=''):

    
    # Reduce vocabulary
    if minimum_count > 0 or max_words > 0:
        print( "Reduce the vocab" )
        reduced_vocab = reduce_vocab(tokens, minimum_count = minimum_count, max_words = max_words)
        #print reduced_vocab.head()
        def clean_data(msg, vocab):
            words_cleaned = filter(lambda word: word not in vocab, msg)
            return words_cleaned
        tokens['error_msg'] = tokens['error_msg'].apply(clean_data, args=[reduced_vocab])
        #print tokens['error_msg'].head()
    
    vocab = get_vocab(tokens)
    
    # Add the average vector
    if avg_vec == True:
        get_avg_vector(tokens, model)
    
    # Encode 
    print( "Encode the messages" )
    
    word2index = {token: token_index for token_index, token in enumerate(list(vocab['word']))}
    
    def encode(msg):
        return [word2index[w] for w in msg]
    tokens['msg_encoded'] = tokens['error_msg'].apply(encode)
    
    embedding_matrix = np.zeros((len(word2index)+1, embedding_dim))
    for word, i in word2index.iteritems():
        embedding_vector = model.wv[word]
        embedding_matrix[i] = model.wv[word]

    # Store
    if store == True:
        np.save(path + 'embedding_matrix_' + name + '.npy', embedding_matrix)
        tokens.to_hdf(path + 'tokens_index_' + name + '.h5', 'frame', mode = 'w')
    
    return word2index, embedding_matrix
    
    

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
    
    
    
    
    
    
