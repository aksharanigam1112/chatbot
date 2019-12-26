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

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
    bag = []
    wrd = [stemmer.stem(w) for w in doc]

    for w in words:
        if(w in wrd):
            bag.append(1)
        else:
            bag.append(0)


    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

