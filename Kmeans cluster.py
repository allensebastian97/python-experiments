# -*- coding: utf-8 -*-


from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from xlrd import open_workbook
import xlwt as wt
from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np








wb = open_workbook('c:/test/input.xlsx',"r")


for s in wb.sheets():
    #print 'Sheet:',s.name
    values = []
    col_value= []
    for row in range(s.nrows):
        for col in range(1):
            value  = (s.cell(row,col).value)
            try : value = str(int(value))
            except : pass
            col_value.append(value)
        
print ( col_value)

t0 = time()

vectorizer = TfidfVectorizer(max_df=0.5, 
                                 min_df=2, 
                                 use_idf=1)

X = vectorizer.fit_transform(col_value)


 
print (vectorizer.get_feature_names())
 km = KMeans(n_clusters=20    , init='k-means++', max_iter=10000, n_init=1,
                verbose=False)

print("Clustering sparse data with %s" % km)
t0 = time()
y=km.fit_predict(X)
print("**************----")

wb1 = wt.Workbook('c:/test/output.xls')
sheet = wb1.add_sheet('output')

    
for row in range(len(y)):
    sheet.write(row, 1, " " + str(y[row]))
    sheet.write(row, 2,col_value[row])
    
    
wb1.save('c:/test/output.xls')

#printing the cluster id and text data            
#for i in range(len(y)):
#    print (str(y[i]) +"    "  + col_value[i])
    
results=[]


results.append(list(km.labels_))
#print (results.shape)
#print (len(results))


print("done in %0.3fs" % (time() - t0))
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
print ("Order_centroids*********")
print(order_centroids.shape)
terms = vectorizer.get_feature_names()
for i in range(4):
   print("Cluster %d:" % i, end='  ')
   for ind in order_centroids[i, :10]:
      print(' %s' % terms[ind], end='')
      print()
