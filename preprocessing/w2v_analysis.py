import pandas as pd
import numpy as np
import gensim
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE  
    

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
    
    
    
    
    
    
