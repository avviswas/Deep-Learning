import pandas as pd
import numpy as np
import gensim as gn
import sys
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords 

data_dir = sys.argv[1]
fake_model = sys.argv[2]
real_model = sys.argv[3]

data = pd.read_csv(data_dir)

data["text"] = data["title"].map(str) + data["text"]
data = data.loc[:,['text','label']]
data['label'] = data['label'].apply(lambda x: 1 if x=='FAKE' else 0)

def remove(text):
  extra_stop_words = ['mr', 'mrs', 'ms','him','her','himself','herself']
  clean = [word for word in text if (word not in (extra_stop_words)) and (len(word) > 1)]
  return clean

data_fake = data.loc[data['label']==1]
data_real = data.loc[data['label']==0]

text_fake = list(data_fake['text'])
text_real = list(data_real['text'])

list_text_fake = []
for article in text_fake:
  list_text_fake.append(remove(list(gn.utils.simple_preprocess(article))))

list_text_real = []
for article in text_real:
  list_text_real.append(remove(list(gn.utils.simple_preprocess(article))))

fake_word2vec = gn.models.Word2Vec(list_text_fake, size = 100, window = 3, min_count = 5, iter = 15)

real_word2vec = gn.models.Word2Vec(list_text_real, size = 100, window = 3, min_count = 5, iter = 15)

print("The 5 Most common Words to 'Hillary' using the built-in Function (Real News):\n\t{}".format(real_word2vec.wv.most_similar(positive='hillary', topn = 5)))

print("The 5 Most common Words to 'Hillary' using the built-in Function (Fake News):\n\t{}".format(fake_word2vec.wv.most_similar(positive='hillary', topn = 5)))


fake_word2vec.save(fake_model)
real_word2vec.save(real_model)
