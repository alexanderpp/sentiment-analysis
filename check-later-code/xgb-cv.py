import xgboost as xgb
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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

vectorizer = CountVectorizer(analyzer = "word",
                             max_features = 3000)

print "Extracting features..."

train_data_features = vectorizer.fit_transform(x)

X = np.array(train_data_features.toarray())
Y = np.array(y)

# dtrain = xgb.DMatrix(X, label=Y)
#
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
#
# res = xgb.cv(param, dtrain, num_boost_round=100, nfold=5,
#              metrics={'error'}, seed = 7,
#              callbacks=[xgb.callback.print_evaluation(show_stdv=True),
#                         xgb.callback.early_stop(3)])
#
#
# print res
# exit(1)

print "Training the model..."

# fit model no training data
model = xgb.XGBClassifier()

kfold = KFold(n_splits=4)
results = cross_val_score(model, X, Y, cv=kfold, verbose=5)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100), results)