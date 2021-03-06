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
from matplotlib import pyplot

from gensim.models import KeyedVectors
from keras.layers.core import Dense, Dropout, SpatialDropout1D, Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers.pooling import GlobalMaxPool1D, MaxPool1D
from keras import Sequential
from keras.regularizers import l1
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras import optimizers
import collections
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_curve,auc, confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from textblob import TextBlob
from langdetect import detect_langs


spanishStemmer=SnowballStemmer("spanish", ignore_stopwords=True)
exclude = set(string.punctuation)
WORD2VECMODEL = "/Users/frandm/Documents/Tesis/Code/SBW-vectors-300-min5.bin.gz"
VOCAB_SIZE = 5000
EMBED_SIZE = 300
NUM_FILTERS = 8
NUM_WORDS = 6
BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_LAYER_SIZE = 32

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
    #return re.sub(r"http\S+", "http://URL", sample)
    return re.sub(r"http\S+", "aquivaunlink", sample)


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
def load_word2vec(vocab_sz):
    word2vec = KeyedVectors.load_word2vec_format(WORD2VECMODEL, binary=True)
    embedding_weights = np.zeros((vocab_sz, EMBED_SIZE))
    for word, index in word2index.items():
        try:
            embedding_weights[index, :] = word2vec[word]
        except KeyError:
            pass
    return embedding_weights

def filter_spanish_only(df):
    spanish_list=list()
    non_spanish_list=list()
    for index, row in df.iterrows():
        text = row['tweets_txt']
        lan = detect_langs(text)
        if lan[0].lang == 'es':
            spanish_list.append(text)
        else:
            non_spanish_list.append(text)
    return spanish_list,non_spanish_list


def stemmer(sample):
    token_words =  [x for x in nltk.word_tokenize(sample)]
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(spanishStemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

pd.set_option('display.max_colwidth', -1)
stop_words = set(stopwords.words('spanish'))
#data load from baseline
c,conn = dblite_connect("/Users/frandm/Documents/Tesis/dataset/Tesis_NLP.sqlite")
df = pd.read_sql_query("SELECT *, 0 as is_ironic FROM baseline_mx_nonironictweets",conn)
df = df.append(pd.read_sql_query("SELECT *, 1 as is_ironic FROM baseline_mx_ironictweets",conn))
#dfbackground = pd.read_sql_query("SELECT *, 0 as is_ironic FROM baseline_mx_background",conn)

#Add more data
#df = df.append(pd.read_sql_query("SELECT tweets_txt, 0 as is_ironic FROM baseline_mx_background",conn))
df = df.append(pd.read_sql_query("SELECT tweets_text as tweets_txt, 1 as is_ironic FROM ling_spain_sat_tweets",conn))
df = df.append(pd.read_sql_query("SELECT tweets_text as tweets_txt, 1 as is_ironic FROM ling_mex_sat_tweets",conn))
df = df.append(pd.read_sql_query("SELECT tweets_text as tweets_txt, 0 as is_ironic FROM ling_spain_nonsat_tweets",conn))
df = df.append(pd.read_sql_query("SELECT tweets_text as tweets_txt, 0 as is_ironic FROM ling_mex_nonsat_tweets",conn))
dfxls_ironi= pd.read_excel ('/Users/frandm/Documents/Tesis/dataset/final_created/Ironi.xlsx')
dfxls_ironi = dfxls_ironi.drop(columns="tweet_id")

dfxls_nonironi= pd.read_excel ('/Users/frandm/Documents/Tesis/dataset/final_created/Non_ironic.xlsx')
dfxls_nonironi = dfxls_nonironi.drop(columns="tweet_id")

print(df.count())

spanish_list,non_spanish_list_iro= filter_spanish_only(dfxls_ironi)
dfxls_ironi=pd.DataFrame(spanish_list, columns=['tweets_txt'])
dfxls_ironi['is_ironic']=1
print(dfxls_ironi)
df = df.append(dfxls_ironi)

print(df.count())
spanish_list,non_spanish_list_non= filter_spanish_only(dfxls_nonironi)
dfxls_nonironi=pd.DataFrame(spanish_list,columns=['tweets_txt'])
dfxls_nonironi['is_ironic']=0
df = df.append(dfxls_nonironi)

print(df.count())

#print(non_spanish_list_iro)
#print(non_spanish_list_non)
#mask = (df['tweets_txt'].str.len() > 50) --didn't get better filtering by lenght
#df=df.loc[mask]

#text pre-processing
#lowercasee
#print(df)
df['tweets_txt']=df['tweets_txt'].str.lower()
#print(df.head())
#change urls
df['tweets_txt']=df.apply(lambda x: change_URL(x['tweets_txt']),axis=1)
#print(df)
#change mentions
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
#df['tweets_txt']=df.apply(lambda x: stemmer(x['tweets_txt']),axis=1)
#embeddings
counter = collections.Counter()
maxlen=0

contawords=0
contachars=0
for index, row in df.iterrows():
    text=row['tweets_txt']
    contawords += len(text.split())
    contachars += len(text)
    words = [x for x in nltk.word_tokenize(text)] #extract each work in the string

    #words = [w for w in words if not w in stop_words]
    if len(words) >  maxlen:
        maxlen = len(words)
        contawords+=len(words)
    for word in words:

        counter[word] +=1

print(contachars / contawords) #average long to be use in charcter embedding
print(contawords / len(df.index)) #average long to be use in charcter embedding

print(counter)
word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]]=wid+1
print(word2index)
vocab_sz = len(word2index) + 1
index2word = {v:k for k,v in word2index.items()}

xs, ys = [], []
for index, row in df.iterrows():
    ys.append(int(row['is_ironic']))
    text = row['tweets_txt']
    words = [x for x in nltk.word_tokenize(text)]  # extract each word in the string
    #words = [w for w in words if not w in stop_words]
    wids = [word2index[word] for word in words]  # extract each work in the string
    xs.append(wids)
X = pad_sequences(xs,maxlen=maxlen)
Y = np_utils.to_categorical(ys)

print(X)
print(Y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,random_state=101)


#load word2vec model
embedding_weights = load_word2vec(vocab_sz)
#embedding_weights=1
model = Sequential()

#if embedding_weights:
model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen, weights=[embedding_weights]))
#else:
#model.add(Embedding(vocab_sz,EMBED_SIZE,input_length=maxlen))

model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=NUM_FILTERS,kernel_size=NUM_WORDS, activation="relu"))
model.add(Conv1D(filters=NUM_FILTERS,kernel_size=NUM_WORDS,activation="relu"))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE,dropout=0.3,recurrent_dropout=0.3,return_sequences=True))
model.add(LSTM(HIDDEN_LAYER_SIZE,dropout=0.3,recurrent_dropout=0.3))
model.add(Dense(8, activation="sigmoid"))
model.add(Dense(2, activation="sigmoid"))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])



# model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])


# checkpoint
filepath="/Users/frandm/Documents/Tesis/weights/words/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
ch = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)


history=model.fit(Xtrain,Ytrain,batch_size=BATCH_SIZE, callbacks=[es,ch], epochs= NUM_EPOCHS, validation_data=(Xtest,Ytest))
score = model.evaluate(Xtest,Ytest,verbose=1)

print("Test Score: {:.3f}, accuracy {:.3f}".format(score[0],score[1]))

#f1-score, precision, recall
y_val_pred= np_utils.to_categorical(model.predict_classes(Xtest)) #one hot enconding to compare with y_yest
print(classification_report(Ytest,y_val_pred)) #f1-score, accuracy

#AUC
y_val_pred= [np.where(r==1)[0][0] for r in y_val_pred]
Ytest_flat= Ytest
Ytest_flat =[np.where(r==1)[0][0] for r in Ytest_flat] #flatenning both to use AUC
false_pos,true_pos,thresholds= roc_curve(Ytest_flat,y_val_pred)
print('AUC Score:{}'.format(auc(false_pos, true_pos)))

#sensitivity, specificity
cm = confusion_matrix(Ytest_flat, y_val_pred)
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print (cm)
print('Sensitivity Score:{}'.format(sensitivity))
print('Specificity Score:{}'.format(specificity))

print('Observations: {}'.format(len(ys)))
print('Ironic share: {}:'.format(ys.count(1)))

pyplot.plot(history.history['acc'],label='train')
pyplot.plot(history.history['val_acc'],label='test')
pyplot.legend()
pyplot.show()








