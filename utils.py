from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.datasets import load_files
import numpy as np


from nltk.tokenize import RegexpTokenizer

import csv

# ----------------------------------------------- Text Manipulation ----------------------------------------------------

def get_sw():
    return stopwords.words('english') + get_stop_words("english")


def get_sentences(text):
    text = text.decode('ascii', 'ignore')

    # try:
    #     text = unicode(text, errors='replace')
    # except:
    #     pass

    return sent_tokenize(text)

def get_words(text):
    text = text.decode('ascii', 'ignore')

    # try:
    #     text = unicode(text, errors='replace')
    # except:
    #     pass

    return word_tokenize(text)

def lemmatize_text(text):
    text = text.decode('utf8', 'ignore')

    # try:
    #     text = unicode(text, errors='replace')
    # except:
    #     pass


    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    tokens = filter(lambda x: x not in get_sw(), tokens)

    for i in range(0, len(tokens)):
        tokens[i] = wordnet_lemmatizer.lemmatize(tokens[i], pos="v");

    result = " ".join(tokens)

    return result.encode('utf-8')

# -------------------------------------------------- Data Loading ------------------------------------------------------

def load_data(file = "./data/labeledTrainData.tsv", test_size = 0.3, seed = 7):
    f = open(file, 'rb')
    reader = csv.reader(f, delimiter="\t")

    first = True

    x = []
    y = []

    for row in reader:
        if first:
            first = False
            continue

        text = row[2]
        category = int(row[1])

        x.append(text)
        y.append(category)

    f.close()

    return train_test_split(x, y, test_size=test_size, random_state=seed)

def load_data2():
    train = load_files("./data/aclImdb/train/")
    x_train, y_train = train.data, train.target

    test = load_files("./data/aclImdb/test/")
    x_test, y_test = test.data, test.target

    return x_train, x_test, y_train, y_test

def extract_features(x, xt, y, yt, vectorizer):
    all_texts = x + xt
    vectorizer.fit(all_texts)

    x = vectorizer.transform(x)
    x = np.array(x.toarray())
    y = np.array(y)

    xt = vectorizer.transform(xt)
    xt = np.array(xt.toarray())
    yt = np.array(yt)

    return x, xt, y, yt