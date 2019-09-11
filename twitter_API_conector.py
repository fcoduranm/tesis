from twython import Twython
from twython import TwythonError
import time
import sqlite3


def dblite_connect(dbname):

    conn = sqlite3.connect(dbname)
    return conn.cursor(),conn

def twitterConnection():
    APP_KEY = 'kKvpeJUkXaPsdC6FmLp12OpZq'
    APP_SECRET = 'CbWCPjpWdnX6yhQ7fThjIzFt2x90VlDvKp0bxoaOaU5Sxxa6C8'
    twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()
    twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)
    return twitter


c,conn = dblite_connect("/Users/frandm/Documents/Tesis/dataset/Tesis_NLP.sqlite")
file = open("/Users/frandm/Documents/Tesis/dataset/Automatic detection of satire in Twitter A psycholinguistic-based approach/SatiricalTwitterDataSet/non-satirical(Mexico).txt","r")
twitter = twitterConnection()

line=file.readline()
count=1
while line:
    #tweet = twitter.show_status(id=tweet_id)
    #print(tweet['text'])
    tweet_id = line

    print(tweet_id)
    try:
        tweet_text = twitter.show_status(id=tweet_id)['text']
        print(tweet_text)
    except TwythonError as e:
        print (e)
        tweet_text= "Tweet not available"

    try:
        c.execute('''INSERT INTO ling_mex_nonsat_tweets (tweet_id,is_ironic,has_image,has_link,has_retweet,tweet_text) VALUES (?,?,?,?,?,?)''', (tweet_id,is_ironic,has_image,has_link,has_retweet,tweet_text))
    except sqlite3.IntegrityError as e:
        print('sqlite error: ', e.args[0])
    line = file.readline()
    count=count+1
    if(count % 700 == 0) :
        del twitter
        conn.commit()
        time.sleep(900)
        twitter = twitterConnection()
    print(count)

file.close()
conn.commit()
conn.close()

