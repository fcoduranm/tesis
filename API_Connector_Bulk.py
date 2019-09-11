from twython import Twython
from twython import TwythonError
import time
import sqlite3
import tweepy
from TwitterAPI import TwitterAPI


def dblite_connect(dbname):

    conn = sqlite3.connect(dbname)
    return conn.cursor(),conn


APP_KEY = 'kKvpeJUkXaPsdC6FmLp12OpZq'
APP_SECRET = 'CbWCPjpWdnX6yhQ7fThjIzFt2x90VlDvKp0bxoaOaU5Sxxa6C8'
auth = tweepy.OAuthHandler(APP_KEY, APP_SECRET)
auth.set_access_token(auth.access_token, auth.access_token_secret)
api = tweepy.API(auth)

c,conn = dblite_connect("/Users/frandm/Documents/Tesis/dataset/Tesis_NLP.sqlite")
#file = open("/Users/frandm/Documents/Tesis/dataset/corpus-ironia-master 2/ironicos.txt","r")
#file = open("/Users/frandm/Documents/Tesis/dataset/corpus-ironia-master 2/noironicos.txt","r")
#file = open("/Users/frandm/Documents/Tesis/dataset/corpus-ironia-master 2/background.txt","r")

#file = open("/Users/frandm/Documents/Tesis/dataset/Automatic detection of satire in Twitter A psycholinguistic-based approach/SatiricalTwitterDataSet/non-satirical(Mexico).txt","r")
#file = open("/Users/frandm/Documents/Tesis/dataset/Automatic detection of satire in Twitter A psycholinguistic-based approach/SatiricalTwitterDataSet/non-satirical(Spain).txt","r")
#file = open("/Users/frandm/Documents/Tesis/dataset/Automatic detection of satire in Twitter A psycholinguistic-based approach/SatiricalTwitterDataSet/Satirical(Mexico).txt","r")
file = open("/Users/frandm/Documents/Tesis/dataset/Automatic detection of satire in Twitter A psycholinguistic-based approach/SatiricalTwitterDataSet/Satirical(Spain).txt","r")

line=file.readline()
nextline=file.readline()
count_ids=1
count_reqs=1
id_list=[]
while line:
    if count_ids % 99 == 0:
        tweets = api.statuses_lookup(id_list)  # id_list is the list of tweet ids
        tweet_txt = []
        for i in tweets:
            tweet_txt.append(i.text)
        print(tweet_txt)
        for i in tweet_txt:
            try:
                #c.execute('''INSERT INTO baseline_mx_ironictweets (tweets_txt) VALUES (?)''',(i,))
                #c.execute('''INSERT INTO baseline_mx_nonironictweets (tweets_txt) VALUES (?)''',(i,))
                c.execute('''INSERT INTO ling_spain_sat_tweets (tweets_text) VALUES (?)''',(i,))
            except sqlite3.IntegrityError as e:
                print('sqlite error: ', e.args[0])
        conn.commit()
        id_list=[]
        count_reqs=count_reqs+1
        count_ids=1
    else:
        #id_list.append(line.split(":")[0]) #works for ironic-nonironic text files
        #id_list.append(line.split('\n')[0]) #for background is just the tweet with line changr special character
        id_list.append(line.split(' ')[1])  # for the linguistic paper there is a space at the beginning of line.
        line = nextline
        nextline = file.readline()
        if not nextline:
            count_ids=99
        else:
            count_ids=count_ids+1
        print(count_ids)
    if count_reqs % 120 == 0:
        time.sleep(900)
        count_reqs=1


file.close()
conn.commit()
conn.close()

