# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import BaseCrossValidator

from nltk.corpus import stopwords
from stop_words import get_stop_words

import csv

f = open("./data/labeledTrainData.tsv", 'rb')
reader = csv.reader(f, delimiter="\t")

first = True

x = []
y = []

print "Loading data ..."

for row in reader:
    if first:
        first = False
        continue

    x.append(row[2])
    y.append(row[1])

f.close()

vectorizer = CountVectorizer(analyzer = "word",
                             max_features = 3000)

print "Extracting features..."

train_data_features = vectorizer.fit_transform(x)

X = np.array(train_data_features.toarray())
Y = np.array(y)


# split data into train and test sets
seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

print "Training the model..."
# fit model no training data
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data

print "Predicting..."

y_pred = model.predict(X_test)

print "Evaluating..."
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))