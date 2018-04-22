import csv
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

lines = [line.rstrip('\n') for line in open('textoutofregdoc.txt')]
#print (lines)

model = Word2Vec(lines, min_count=1)
#print (model)

words = model[model.wv.vocab]
#print(words)

print(model.wv.vocab)

model.save('model.bin')
new_model = Word2Vec.load('model.bin')
#print(new_model)


pca = PCA(n_components=2)
result = pca.fit_transform(words)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
