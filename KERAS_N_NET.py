
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
#import time
#import matplotlib.pyplot as plt


dataframe_all = pd.read_csv("/home/allen/Desktop/CODE/weather.csv")
# print(dataframe_all['WindDir9am'])
dataframe_all = dataframe_all.replace(['NaN','N', 'NNE', 'NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
dataframe_all = dataframe_all.replace(['No','Yes'],[0,1])
dataframe_all = dataframe_all.drop('Date',1)
dataframe_all = dataframe_all.drop('Location',1)
dataframe_all= dataframe_all.dropna(axis=0, how='any')
y=dataframe_all['RainTomorrow']
x=dataframe_all.drop('RainTomorrow',1)

#print(x)
#convert dataframe into arrays
x=x.as_matrix()
y=y.astype(str).str.strip('[]').str.get_dummies(', ')
y=y.as_matrix()
x=np.asarray(x)
y=np.asarray(y)
#plt.plot(x)
#print(x)
print(y)

np.random.seed(42)
model = Sequential()
model.add(Dense(7,input_shape=(21,),activation='tanh'))
model.add(Dense(3,activation='tanh'))
model.add(Dense(2,activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(x,y,batch_size=100,verbose=2,epochs=100)

print("\n\n\n<----------------------------->\n\n\n")
scores = model.evaluate(x, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))




