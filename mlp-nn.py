import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

from nltk.corpus import stopwords
from stop_words import get_stop_words

import sys
sys.path.append('.')
import utils


max_features = 3000

print "Loading data ..."
x_train, x_test, y_train, y_test = utils.load_data(file="./data/labeledTrainData.tsv")

sw = stopwords.words('english') + get_stop_words("english")

print "Extracting features..."
vectorizer = CountVectorizer(analyzer = "word",
                             max_features = 3000)

x_train, x_test, y_train, y_test = utils.extract_features(x_train, x_test, y_train, y_test, vectorizer)

print "Creating and training the model..."

model = MLPClassifier(solver='sgd',
                      hidden_layer_sizes=(10, 10, 10),
                      random_state=5,
                      verbose=True)

model.fit(x_train, y_train)

print "Predicting..."

y_pred = model.predict(x_test)

print "Evaluating..."
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))