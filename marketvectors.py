import numpy as np
import pandas as pd
import datetime
from keras.optimizers import SGD
from keras.datasets.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Dropout
from keras.layers import recurrent as rnn
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import PReLU
import datetime
from datetime import datetime
#import operator
import pandas.io.data
import re
from dateutil import parser
from pprint import pprint
import time
import six.moves.cPickle as pickle
#from backtest import Strategy, Portfolio


X_market = np.load('allstockdata.npy')
X_text = np.load('X-text-final.npy')
X_sent = np.load('X_sent.npy')
Y_cat = np.load('Y-correct.npy')

#print("X shape is",X.shape)
print("X_text shape is",X_text.shape)
print("X_market shape is", X_market.shape)
#print("testX shape is",testX.shape)
#print("Y shape is",Y.shape)
#print("testY shape is", len(testY))

vocab_size = np.amax(X_text) + 1
print("vocab size is", vocab_size)


#Y_cat = np_utils.to_categorical(Y, 2)
#testY_cat = np_utils.to_categorical(Y,2)

#print ("Y_cat shape is",Y_cat.shape)

newsnetwork = Sequential()
newsnetwork.add(Embedding(vocab_size,128, mask_zero=True))
newsnetwork.add(rnn.JZS3(100, return_sequences=False))
newsnetwork.add(Dropout(0.5))

marketnetwork = Sequential()
marketnetwork.add(GaussianNoise(.01,input_shape=(5, 70)))
marketnetwork.add(rnn.GRU(100,input_dim=70))
#marketnetwork.add(BatchNormalization())


#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sentnetwork = Sequential()
sentnetwork.add(Dense(100,input_dim=10,activation="softmax"))
#sentnetwork.add(BatchNormalization())
sentnetwork.add(PReLU())
#sentnetwork.add(Dropout(0.75))


model = Sequential()
model.add(Merge([newsnetwork,marketnetwork,sentnetwork],mode='concat'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
model.fit([X_text, X_market, X_sent], Y_cat, batch_size=100, nb_epoch=100, validation_split=0.50, show_accuracy=True)
#loss, acc = model.evaluate(testX, testY, batch_size=127, show_accuracy=True)
#print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))





















#nyturl = http://api.nytimes.com/svc/search/v2/articlesearch.json?

#query = nyturl + 'q=wall street&fq=news_desk:("Business" "Financial" "Your Money")&f1=abstract&in_date=20051101&end_date=20051101&api-key=' + apikey
