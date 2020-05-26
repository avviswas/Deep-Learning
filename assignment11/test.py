import pandas as pd
import numpy as np
import gensim as gn
from gensim.models import Word2Vec
import sys
from numpy import dot
from numpy.linalg import norm


model_fname = sys.argv[1]
model_rname = sys.argv[2]
query_dir = sys.argv[3]

fword2vect = Word2Vec.load(model_fname)
frword2vect = Word2Vec.load(model_rname)


f = open(query_dir)
l = f.readline()
query_words = []
while(l != ""):
  l = l.split()
  query_words.append(l[0].lower())
  l = f.readline()
f.close()

def similar(model, query, topNumber):
  cosineSimilarity = {}
  vocab = list(model.wv.vocab)
  a = model[query]
  normalize = norm(a)
  for v in vocab:
    if v != query:
      b = model[v]
      sim = np.dot(a,b)/(normalize*norm(b))
      cosineSimilarity[v] = sim
  cosineSimilarity = sorted(cosineSimilarity.items(), key = lambda dist: dist[1], reverse = True)
  mostSimilarity = []
  i = 0;
  for item in cosineSimilarity:
    mostSimilarity.append((item[0], item[1]))
    i += 1
    if i == topNumber:
      break
  return mostSimilarity

# Normalize Vectors
fword2vect.init_sims(replace = True)
frword2vect.init_sims(replace = True)

print("The Query Words are:\t{}".format(query_words))

print("Fake News Data -> Top 5 Similar Words are")
for words in query_words:
  similar_words = similar(frword2vect, words, 5)
  print(words)
  for sim_word in similar_words:
    print("\t{}".format(sim_word))
print("\nReal News Data -> Top 5 Similar Words are")
for words in query_words:
  similar_words = similar(frword2vect, words, 5)
  print(words)
  for sw in similar_words:
    print("\t{}".format(sw))
