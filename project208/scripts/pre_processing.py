import nltk
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def stopwords_removal(text):
    wh_words = ['my', 'your', 'his', 'her', 'this', 'do', "don't"]
    stop = set(stopwords.words('english'))
    for word in wh_words:
        stop.remove(word)
    tweet = ' '.join([x for x in text.split() if x not in stop])
    return tweet


def lemmatize(corpus):
    lem = WordNetLemmatizer()
    #corpus = [lem.lemmatize(x, pos = 'v') for x in corpus]
    corpus = [lem.lemmatize(x, pos = 'r') for x in corpus]
    corpus = [lem.lemmatize(x, pos = 'n') for x in corpus]
    return corpus


def stem(corpus, stem_type = None):
    stemmer = SnowballStemmer(language='english')
    corpus = [stemmer.stem(x) for x in corpus]
    return corpus


def pre_processing(tweet):
    """
    input: a string
    output: tokenized and preprocessed text (a list)
    """

    tweet = tweet.lower()

    #changing urls with a unique word
    p1 = re.compile(r'\bhttps://t.co/\w+')
    p2 = re.compile(r'\bhttps://^(t.co/\w+)\w')
    tweet = re.sub(p1, 'tweeturl', tweet)
    tweet = re.sub(p2, 'simpleurl', tweet)

    #changing all spaces with a simple space and dots and commas
    p3 = re.compile(r'\s+')
    p4 = re.compile(r"[.,;'’-]")
    tweet = re.sub(p4, '', tweet)
    tweet = re.sub(p3, ' ', tweet)


    #removing stopwords
    tweet = stopwords_removal(tweet)

    #tokenize the tweet
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tweet = tokenizer.tokenize(tweet)
    tweet = remove_repeated_characters(tweet)

    #lemmatization and stemming
    tweet = lemmatize(tweet)
    tweet = stem(tweet)
    
    return tweet


def bag_of_words(tweets):
    """
    input:  a series of tokenized tweets (lists)
    output: a vocabulary (list) and a bag of words (numpy array)
    """

    set_of_words = set()
    for tweet in tweets:
        for word in tweet:
            set_of_words.add(word)
    vocab = list(set_of_words)

    position = {}
    for i, token in enumerate(vocab):
        position[token] = i

    bow_matrix = np.zeros((len(tweets), len(vocab)))

    for i, preprocessed_sentence in enumerate(tweets):
        for token in preprocessed_sentence:   
            bow_matrix[i][position[token]] = bow_matrix[i][position[token]] + 1

    return vocab, bow_matrix


def oversampler(dict_of_words, noise):
    """
    input: dict of words is a dictionary key: part of sentence - value: a dataframe of tokens and their relative frequency
            noise is a series of relative frequencies, indexed by tokens
    output: a random string
    """

    adj = np.random.choice(dict_of_words['a'].tokens, 1, p=dict_of_words['a'].freq)
    noun = np.random.choice(dict_of_words['n'].tokens, 3, p=dict_of_words['n'].freq)
    verb = np.random.choice(dict_of_words['v'].tokens, 2, p=dict_of_words['v'].freq)
    extra = np.random.choice(dict_of_words['r'].tokens, 3, p=dict_of_words['r'].freq)
    noise = np.random.choice(noise.index, 3, p=noise.values)

    
    return ' '.join([noun[0],extra[0],verb[0],noun[1],extra[1],adj[0],noun[2],verb[1],extra[2]\
        ,noise[0],noise[1],noise[2]]
        )