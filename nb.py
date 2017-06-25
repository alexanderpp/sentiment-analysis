from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.metrics import accuracy_score
import sys
sys.path.append('.')
import utils

print "Loading data..."
x_train, x_test, y_train, y_test = utils.load_data()
# x_train, x_test, y_train, y_test = utils.load_data2()

sw = stopwords.words('english') + get_stop_words("english")

print "Extracting features..."
vectorizer = CountVectorizer(analyzer = "word",
                             max_df=0.5,
                             min_df=50,
                             stop_words=sw,
                             max_features = 250)

x_train, x_test, y_train, y_test = utils.extract_features(x_train, x_test, y_train, y_test, vectorizer)

model = GaussianNB()

print "Training the model..."
model.fit(x_train, y_train)

print "Predicting..."
y_pred = model.predict(x_test)

print "Evaluating..."
accuracy = accuracy_score(y_test, y_pred.round())
print("Accuracy: %.2f%%" % (accuracy * 100.0))
