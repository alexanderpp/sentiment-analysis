from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from stop_words import get_stop_words

import csv

f = open("./data/labeledTrainData.tsv", 'rb')
reader = csv.reader(f, delimiter="\t")

first = True

pre_x = []
pre_y = []

print "Loading data ..."

for row in reader:
    if first:
        first = False
        continue

    pre_x.append(row[2])
    pre_y.append(row[1])

f.close()

total_data_size = len(pre_y)
training_data_size = int(total_data_size * 0.90)


x = pre_x[:training_data_size]
test_x = pre_x[training_data_size:]

y = pre_y[:training_data_size]
test_y = pre_y[training_data_size:]

sw = stopwords.words('english') + get_stop_words("english")

vectorizer = CountVectorizer(analyzer = "word",
                             max_df=0.5,
                             min_df=50,
                             stop_words=sw,
                             max_features = 250)

print "Extracting features..."

train_data_features = vectorizer.fit_transform(x)
test_data_features = vectorizer.fit_transform(test_x)

x = np.array(train_data_features.toarray())
y = np.array(y)

test_x = np.array(test_data_features.toarray())

# Create a Gaussian Classifier
model = GaussianNB()

print "features", vectorizer.get_feature_names()
print "stop words", vectorizer.get_stop_words()

print "Training the model..."

# Train the model using the training sets
model.fit(x, y)

print "Predicting and evaluating..."

# Predict Output
predicted = model.predict(test_x)


count = 0
for i, v in enumerate(predicted):
    if v == test_y[i]:
        count = count + 1


print "Result: ", count/float(len(predicted))*100, "%"
