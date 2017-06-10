import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

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
    y.append(int(row[1]))

f.close()

sw = stopwords.words('english') + get_stop_words("english")

vectorizer = CountVectorizer(analyzer = "word",
                             max_features = 1000)

print "Extracting features..."

train_data_features = vectorizer.fit_transform(x)

X = np.array(train_data_features.toarray())
Y = np.array(y)

seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


print "Creating and training the model..."

model = MLPClassifier(solver='sgd',
                      hidden_layer_sizes=(5, 2),
                      random_state=5)

model.fit(X_train, y_train)

print "Predicting..."

y_pred = model.predict(X_test)

print "Evaluating..."
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))