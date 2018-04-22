# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:41:25 2018

@author: allens
"""

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
model = NMF(n_components=num_topics, init='nndsvd');




#fit the model
model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict)

get_nmf_topics(model, 20)