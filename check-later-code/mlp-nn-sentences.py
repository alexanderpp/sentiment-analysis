import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append('.')

import utils

print "Loading data ..."
_x_train, x_test, _y_train, y_test = utils.load_data(file="./data/labeledTrainData-lemmatized.tsv")

x_train = []
y_train = []

for i in range(0, len(_x_train)):
    sentences = utils.get_sentences(_x_train[i])

    for sentence in sentences:
        x_train.append(sentence)
        y_train.append(_y_train[i])


del _x_train, _y_train

# sw = stopwords.words('english') + get_stop_words("english")

vectorizer = CountVectorizer(analyzer = "word",
                             max_features = 3000)

print "Extracting features..."

train_data_features = vectorizer.fit_transform(x_train)

x_train = np.array(train_data_features.toarray())
y_train = np.array(y_train)

del train_data_features

print "Creating and training the model..."
model = MLPClassifier(solver='sgd',
                      hidden_layer_sizes=(10, 10, 10),
                      verbose=True)

model.fit(x_train, y_train)

print "Predicting..."
y_pred = []

for i in range(0, len(x_test)):
    sentences = utils.get_sentences(x_test[i])

    correct = 0
    total = 0

    sentence_vectors = vectorizer.transform(sentences)
    sentence_vectors = np.array(sentence_vectors.toarray())

    pred = model.predict(sentence_vectors)

    pred_score = sum(pred) * 1.0 / len(pred)

    pred = int(round(pred_score))

    y_pred.append(pred)


print "Evaluating..."
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))