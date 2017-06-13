from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import csv

# ----------------------------------------------- Text Manipulation ----------------------------------------------------

def get_sentences(text):
    return sent_tokenize(unicode(text, errors='replace'))

def get_words(text):
    return word_tokenize(unicode(text, errors='replace'))

def lemmatize_text(text):
    tokens = get_words(unicode(text, errors='replace'))

    for i in range(0, len(tokens)):
        tokens[i] = wordnet_lemmatizer.lemmatize(tokens[i], pos="v");

    return " ".join(tokens)

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