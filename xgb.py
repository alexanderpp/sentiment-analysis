import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
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
                             max_features = max_features)

x_train, x_test, y_train, y_test = utils.extract_features(x_train, x_test, y_train, y_test, vectorizer)

print "Training the model..."
xgb_params = {
    'objective': 'binary:logistic',
    #'colsample_bytree': 0.8,
    'silent':1,
    'subsample': 0.8,
    'learning_rate': 0.5,
    'max_depth': 8,
    'num_parallel_tree': 1,
    'min_child_weight': 10,
    'eval_metric': 'auc',
    'seed':0
}

dtrain = xgb.DMatrix(x_train, label=y_train)
watchlist  = [(dtrain,'train')]
model = xgb.train(xgb_params, dtrain, 128, watchlist)

print "Predicting..."
dtest = xgb.DMatrix(x_test)
y_pred = model.predict(dtest)

print "Evaluating..."
accuracy = accuracy_score(y_test, y_pred.round())
print("Accuracy: %.2f%%" % (accuracy * 100.0))