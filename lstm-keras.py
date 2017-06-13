
from keras.preprocessing import sequence
from keras.preprocessing import text as text_processing

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import keras

print('Loading data...')

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

# sw = stopwords.words('english') + get_stop_words("english")

print "Extracting features..."

max_features = 20000
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

train_data_features = [text_processing.one_hot(i, max_features) for i in x]

X = np.array(train_data_features)
Y = np.array(y)

seed = 7
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

del X, Y, train_data_features

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))

score, accuracy = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

"""
Best: 83.45%

With maxlen 80 -> 79.38%
With maxlen 500 -> 83.45%

17500/17500 [==============================] - 332s - loss: 0.5189 - acc: 0.7467 - val_loss: 0.5170 - val_acc: 0.7647
Epoch 2/15
17500/17500 [==============================] - 325s - loss: 0.3673 - acc: 0.8453 - val_loss: 0.4733 - val_acc: 0.7833
Epoch 3/15
17500/17500 [==============================] - 321s - loss: 0.2651 - acc: 0.8962 - val_loss: 0.4070 - val_acc: 0.8433
Epoch 4/15
17500/17500 [==============================] - 323s - loss: 0.2205 - acc: 0.9153 - val_loss: 0.4620 - val_acc: 0.8208
Epoch 5/15
17500/17500 [==============================] - 319s - loss: 0.1752 - acc: 0.9349 - val_loss: 0.4509 - val_acc: 0.8472
Epoch 6/15
17500/17500 [==============================] - 319s - loss: 0.1474 - acc: 0.9457 - val_loss: 0.5115 - val_acc: 0.8359
Epoch 7/15
17500/17500 [==============================] - 318s - loss: 0.1151 - acc: 0.9582 - val_loss: 0.5179 - val_acc: 0.8367
Epoch 8/15
17500/17500 [==============================] - 323s - loss: 0.0962 - acc: 0.9651 - val_loss: 0.5906 - val_acc: 0.8219
Epoch 9/15
17500/17500 [==============================] - 317s - loss: 0.0593 - acc: 0.9798 - val_loss: 0.7817 - val_acc: 0.8347
Epoch 10/15
17500/17500 [==============================] - 317s - loss: 0.0461 - acc: 0.9846 - val_loss: 0.7029 - val_acc: 0.8355
Epoch 11/15
17500/17500 [==============================] - 316s - loss: 0.0413 - acc: 0.9858 - val_loss: 0.8405 - val_acc: 0.8127
Epoch 12/15
17500/17500 [==============================] - 322s - loss: 0.0286 - acc: 0.9905 - val_loss: 0.9727 - val_acc: 0.8148
Epoch 13/15
17500/17500 [==============================] - 318s - loss: 0.0318 - acc: 0.9895 - val_loss: 0.8785 - val_acc: 0.8243
Epoch 14/15
17500/17500 [==============================] - 317s - loss: 0.0259 - acc: 0.9921 - val_loss: 0.8331 - val_acc: 0.8215
Epoch 15/15
17500/17500 [==============================] - 317s - loss: 0.0172 - acc: 0.9951 - val_loss: 0.9272 - val_acc: 0.8345
7500/7500 [==============================] - 38s     
('Test score:', 0.92720497883955633)
Accuracy: 83.45%


"""