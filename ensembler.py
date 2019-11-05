import time
import sqlite3
import csv
import pandas as pd
import re
import nltk
import string
import collections
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt')
from nltk.stem import SnowballStemmer
from matplotlib import pyplot
from keras.models import load_model
import scipy

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
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
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
import statistics


spanishStemmer=SnowballStemmer("spanish", ignore_stopwords=True)
exclude = set(string.punctuation)
WORD2VECMODEL = "/Users/frandm/Documents/Tesis/Code/SBW-vectors-300-min5.bin.gz"

THRESHOLD = 15
VOCAB_SIZE = 5000

stop_words = set(stopwords.words('spanish'))

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
    return re.sub(r"http\S+", "aquivaunaURL", sample)


def change_mentions(sample):
    return re.sub(r"@\S+", "@", sample)

def remove_whitespaces(sample):
    return re.sub(r"\s+", " ", sample)

def change_numbers(sample):
    return ''.join([i for i in sample if not i.isdigit()])

def remove_stopwords(sample):
    return ''.join([w for w in sample if not w in stop_words])

def remove_punctuations(sample):
    return ''.join(i for i in sample if i not in exclude)

def remove_hashtags(sample):
    sample = sample.replace("\n", "")
    sample = sample.replace("#sarcasmo","")
    sample = sample.replace("#ironia", "")
    sample = sample.replace("#ironía", "")
    sample = sample.replace("#", "")
    return sample
def stemmer(sample):
    token_words =  [x for x in nltk.word_tokenize(sample)]
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(spanishStemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

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


def data_preparation():
    pd.set_option('display.max_colwidth', -1)

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
    #df['tweets_txt']=df.apply(lambda x: remove_stopwords(x['tweets_txt']),axis=1)
    df['tweets_txt']=df.apply(lambda x: stemmer(x['tweets_txt']),axis=1)
    print(df)
    #stopwords?
    return df

def char_data_prep(df):
    counter = collections.Counter()
    maxlen = 0

    print(df['tweets_txt'].apply(lambda x: len(x)).describe())

    unique_symbols = Counter()

    for _, message in df['tweets_txt'].iteritems():
        unique_symbols.update(message)

    print("Unique symbols:", len(unique_symbols))

    print("Unique symbols:", unique_symbols)

    uncommon_symbols = list()

    for symbol, count in unique_symbols.items():
        if count < THRESHOLD:
            uncommon_symbols.append(symbol)

    print("Uncommon symbols:", len(uncommon_symbols))
    print("Uncommon symbols:", uncommon_symbols)

    DUMMY = uncommon_symbols[0]
    tr_table = str.maketrans("".join(uncommon_symbols), DUMMY * len(uncommon_symbols))

    df['tweets_txt'] = df['tweets_txt'].apply(lambda x: x.translate(tr_table))

    # We will need the number of unique symbols further down when we will decide on the dimensionality of inputs.

    num_unique_symbols = len(unique_symbols) - len(uncommon_symbols) + 1

    tokenizer = Tokenizer(
        char_level=True,
        filters=None,
        lower=False,
        num_words=num_unique_symbols
    )

    print(df)

    tokenizer.fit_on_texts(df['tweets_txt'])
    sequences = tokenizer.texts_to_sequences(df['tweets_txt'])
    X = pad_sequences(sequences, maxlen=289)
    Y = np_utils.to_categorical(df['is_ironic'])

    print(sequences[0])
    print(X[0])
    print(Y[0])

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,random_state=101)
    return Xtrain, Xtest, Ytrain, Ytest

def words_data_prep(df):
    counter = collections.Counter()
    maxlen = 0

    contawords = 0
    contachars = 0
    for index, row in df.iterrows():
        text = row['tweets_txt']
        contawords += len(text.split())
        contachars += len(text)
        words = [x for x in nltk.word_tokenize(text)]  # extract each work in the string

        # words = [w for w in words if not w in stop_words]
        if len(words) > maxlen:
            maxlen = len(words)
            contawords += len(words)
        for word in words:
            counter[word] += 1

    print(contachars / contawords)  # average long to be use in charcter embedding
    print(contawords / len(df.index))  # average long to be use in charcter embedding

    print(counter)
    word2index = collections.defaultdict(int)
    for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
        word2index[word[0]] = wid + 1
    print(word2index)
    vocab_sz = len(word2index) + 1
    index2word = {v: k for k, v in word2index.items()}

    xs, ys = [], []
    for index, row in df.iterrows():
        ys.append(int(row['is_ironic']))
        text = row['tweets_txt']
        words = [x for x in nltk.word_tokenize(text)]  # extract each word in the string
        # words = [w for w in words if not w in stop_words]
        wids = [word2index[word] for word in words]  # extract each work in the string
        xs.append(wids)
    X = pad_sequences(xs, maxlen=maxlen)
    Y = np_utils.to_categorical(ys)

    print(X)
    print(Y)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=101)
    return  Xtrain, Xtest, Ytrain, Ytest



df=data_preparation()
XtrainC, XtestC, YtrainC, YtestC = char_data_prep(df)
XtrainW, XtestW, YtrainW, YtestW = words_data_prep(df)

print(XtestC)
print(XtestW)
print(XtestC)


model_words = load_model("/Users/frandm/Documents/Tesis/weights/words/weights-improvement-04-0.74.hdf5")

model_chars = load_model("/Users/frandm/Documents/Tesis/weights/weights-improvement-11-0.73.hdf5")

#score = model_words.evaluate(XtestW,YtestW,verbose=1)
#print("Test Score W: {:.3f}, accuracy {:.3f}".format(score[0],score[1])x


#score = model_chars.evaluate(XtestC,YtestC,verbose=1)
#print("Test Score C: {:.3f}, accuracy {:.3f}".format(score[0],score[1]))

y_val_predW= model_words.predict_classes(XtestW) #one hot enconding to compare with y_yest
print("Words:")
print(classification_report(YtestW,np_utils.to_categorical(y_val_predW)))

#print(classification_report(YtestW,y_val_predW))

y_val_predC= model_chars.predict_classes(XtestC) #one hot enconding to compare with y_yest
print("Chars:")
print(classification_report(YtestC,np_utils.to_categorical(y_val_predC)))


#print(classification_report(YtestC,y_val_predC))


ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(df['tweets_txt'])
X = ngram_vectorizer.transform(df['tweets_txt'])
Y = df['is_ironic']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,random_state=101)

svm = LinearSVC(class_weight="balanced",verbose=10)
#svm = SGDClassifier(class_weight="balanced", average=True, alpha=1e-4, tol=1e-5, verbose=10)
svm.fit(Xtrain, Ytrain)
y_val_svm=svm.predict(Xtest)
print("SVM:")
print(classification_report(Ytest,y_val_svm))

final_pred= []
for i in range(0,len(XtestW)):
    final_pred.append(statistics.mode([y_val_predW[i], y_val_predC[i],y_val_svm[i]]))
y_val_pred= np_utils.to_categorical(final_pred)
print("SVM:")
print(classification_report(YtestC,y_val_pred))


