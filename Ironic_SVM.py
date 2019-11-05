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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier



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


ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(df['tweets_txt'])
X = ngram_vectorizer.transform(df['tweets_txt'])
Y = df['is_ironic']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,random_state=101)

svm = LinearSVC(class_weight="balanced",verbose=10)
#svm = SGDClassifier(class_weight="balanced", average=True, alpha=1e-4, tol=1e-5, verbose=10)
svm.fit(Xtrain, Ytrain)
print(classification_report(Ytest,svm.predict(Xtest)))








