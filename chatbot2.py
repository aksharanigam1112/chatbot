import nltk
from nltk.stem import LancasterStemmer
import numpy as np
import json
# import tensorflow
# import tflearn
import random
import warnings
warnings.filterwarnings(action="ignore")

stemmer= LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent["patterns"]:
        wrd = nltk.word_tokenize(pattern)
        words.extend(wrd)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if(intent["tag"] not in labels):
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
labels =sorted(labels)

