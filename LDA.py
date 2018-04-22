# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:46:25 2018

@author: allens
"""
import sys
#import logging
import pandas as pd
#import numpy as np
#from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
#from sklearn.externals import joblib


#import scipy as sp;
#import sklearn;
  
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
 
 
from gensim.models import ldamodel
import gensim.corpora;
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
 
#from sklearn.preprocessing import normalize;
import pickle;
#from textacy import viz
#
#LOGGER = logging.getLogger(__name__)
#
#model = textacy.tm.TopicModel('nmf', n_topics=20)
#model.fit(doc_term_matrix)
#model
## TopicModel(n_topics=10, model=NMF)

#from nltk.corpus import stopwords;
#import nltk;
from sklearn.feature_extraction import text
stopwords = text.ENGLISH_STOP_WORDS


data = pd.read_csv('C:/Users/allens/final_comments.csv', error_bad_lines=False,encoding="ISO-8859-1",header=None,names=['headline_text'])    

#data  = data.astype('str')
#data.column
#data['headline_text']=data!
for idx in range(len(data)):
    
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    data.iloc[idx]['headline_text'] = [word for word in data.iloc[idx]['headline_text'].split(' ') 
    if word not in stopwords];
    
    #print logs to monitor output
    if idx % 1000 == 0:
        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data)));

#pickle.dump(data, open('data_text.dat', 'wb'))

train_headlines = [value[0] for value in data.iloc[0:].values];
#print(train_headlines)
num_topics = 20


#id2word = gensim.corpora.Dictionary(train_headlines);
##print(id2word) 
#corpus = [id2word.doc2bow(text) for text in train_headlines];
##print(corpus)  
#lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics);
#
#def get_lda_topics(model, num_topics):
#    word_dict = {};
#    for i in range(num_topics):
#        words = model.show_topic(i, topn = 20);
#        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
#    return pd.DataFrame(word_dict);
#
#a=get_lda_topics(lda, num_topics)
#print(type(a))
""""""""""""""""""""""""""""""""""""""""""""""""
train_headlines_sentences = [' '.join(text) for text in train_headlines]


# Now, we obtain a Counts design matrix, for which we use SKLearn’s CountVectorizer module. The transformation will return a matrix of size (Documents x Features), where the value of a cell is going to be the number of times the feature (word) appears in that document.
# 
# To reduce the size of the matrix, to speed up computation, we will set the maximum feature size to 5000, which will take the top 5000 best features that can contribute to our model.



vectorizer = CountVectorizer(analyzer='word', max_features=5000);
x_counts = vectorizer.fit_transform(train_headlines_sentences);


# Next, we set a TfIdf Transformer, and transform the counts with the model.



transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);


# And now we normalize the TfIdf values to unit length for each row.



xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)


# And finally, obtain a NMF model, and fit it with the sentences.



#obtain a NMF model.
model_nmf = NMF(n_components=num_topics, init='nndsvd');




#fit the model
model_nmf.fit(xtfidf_norm)

def get_nmf_topics(model_nmf, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model_nmf.components_[i].argsort()[:-200 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict)

NMF=get_nmf_topics(model_nmf, 200)
print(NMF)