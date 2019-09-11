import time
import sqlite3
import csv
import pandas as pd
import re
import nltk
import string
import collections
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.stem import SnowballStemmer

from gensim.models import KeyedVectors
from keras.layers.core import Dense, Dropout, SpatialDropout1D, Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Flatten
from keras.layers.pooling import GlobalMaxPool1D, MaxPool1D
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import numpy as np


spanishStemmer=SnowballStemmer("spanish", ignore_stopwords=True)
exclude = set(string.punctuation)
WORD2VECMODEL = "/Users/frandm/Documents/Tesis/Code/SBW-vectors-300-min5.bin.gz"
VOCAB_SIZE = 5000
EMBED_SIZE = 300
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 20
HIDDEN_LAYER_SIZE = 256

def dblite_connect(dbname):

    conn = sqlite3.connect(dbname)
    return conn.cursor(),conn

def csvwrite(csvname,input): #problemas con caracteres en espanio como tildes
    with open(csvname, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in input:
            csvrow =[row,'0']
            csvwriter.writerow(csvrow)
    csvfile.close()
    return

def change_URL(sample):
    return re.sub(r"http\S+", "http://URL", sample)


def change_mentions(sample):
    return re.sub(r"@\S+", "@", sample)

def remove_whitespaces(sample):
    return re.sub(r"\s+", " ", sample)

def change_numbers(sample):
    return ''.join([i for i in sample if not i.isdigit()])

def remove_punctuations(sample):
    return ''.join(i for i in sample if i not in exclude)

def remove_hashtags(sample):
    sample = sample.replace("\n", "")
    sample = sample.replace("#sarcasmo","")
    sample = sample.replace("#ironia", "")
    sample = sample.replace("#ironía", "")
    sample = sample.replace("#", "")
    return sample
def load_word2vec():
    word2vec = KeyedVectors.load_word2vec_format(WORD2VECMODEL, binary=True)
    embedding_weights = np.zeros((vocab_size, EMBED_SIZE))
    for word, index in word2index.items():
        try:
            embedding_weights[index, :] = word2vec[word]
        except KeyError:
            pass
    return embedding_weights

pd.set_option('display.max_colwidth', -1)
stop_words = set(stopwords.words('spanish'))
#data load from baseline
c,conn = dblite_connect("/Users/frandm/Documents/Tesis/dataset/Tesis_NLP.sqlite")
df = pd.read_sql_query("SELECT *, 0 as is_ironic FROM baseline_mx_nonironictweets",conn)
df = df.append(pd.read_sql_query("SELECT *, 1 as is_ironic FROM baseline_mx_ironictweets",conn))
#dfbackground = pd.read_sql_query("SELECT *, 0 as is_ironic FROM baseline_mx_background",conn)

#text pre-processing
#lowercasee
#print(df)
df['tweets_txt']=df['tweets_txt'].str.lower()
#print(df.head())
#change urls
df['tweets_txt']=df.apply(lambda x: change_URL(x['tweets_txt']),axis=1)
#print(df)
#chane mentions
df['tweets_txt']=df.apply(lambda x: change_mentions(x['tweets_txt']),axis=1)
#print(df)
#remove# and #sarcasmo
df['tweets_txt']=df.apply(lambda x: remove_hashtags(x['tweets_txt']),axis=1)
#print(df)

df['tweets_txt']=df.apply(lambda x: change_numbers(x['tweets_txt']),axis=1)
print(df)
#df['tweets_txt']=df.apply(lambda x: remove_punctuations(x['tweets_txt']),axis=1)
print(df)
df['tweets_txt']=df.apply(lambda x: remove_whitespaces(x['tweets_txt']),axis=1)
print(df)
#stopwords?

#embeddings
counter = collections.Counter()
maxlen=0
for index, row in df.iterrows():
    text=row['tweets_txt']
    #print(text)
    words = [x for x in nltk.word_tokenize(text)] #extract each work in the string
    #words = [w for w in words if not w in stop_words]
    if len(words) >  maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] +=1

print(counter)
word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]]=wid+1
print(word2index)
vocab_size = len(word2index) + 1
index2word = {v:k for k,v in word2index.items()}

xs, ys = [], []
for index, row in df.iterrows():
    ys.append(int(row['is_ironic']))
    words = [x for x in nltk.word_tokenize(text)]  # extract each work in the string
    #words = [w for w in words if not w in stop_words]
    wids = [word2index[word] for word in words]  # extract each work in the string
    xs.append(wids)
X = pad_sequences(xs,maxlen=maxlen)
Y = np_utils.to_categorical(ys)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)


#load word2vec model
#embedding_weights = load_word2vec()
embedding_weights=0
model = Sequential()

if embedding_weights !=0:
    model.add(Embedding(vocab_size, EMBED_SIZE, input_length=maxlen, weights=[embedding_weights]))
else:
    model.add(Embedding(vocab_size,EMBED_SIZE,input_length=maxlen))

#model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=NUM_FILTERS,kernel_size=NUM_WORDS))
model.add(Activation("sigmoid"))
model.add(BatchNormalization())
model.add(Conv1D(filters=NUM_FILTERS,kernel_size=NUM_WORDS))
model.add(Activation("sigmoid"))
#model.add(Dropout(0.5))
model.add(LSTM(HIDDEN_LAYER_SIZE,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(HIDDEN_LAYER_SIZE,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(2, activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
history=model.fit(Xtrain,Ytrain,batch_size=BATCH_SIZE, epochs= NUM_EPOCHS, validation_data=(Xtest,Ytest))
score = model.evaluate(Xtest,Ytest,verbose=1)
print("Test Score: {:.3f}, accuracy {:.3f}".format(score[0],score[1]))






