import gensim
from gensim.utils import tokenize
import numpy as np
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential

print('Loading w2v...')
w2v = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

timesteps = 400
dimensions = 300
batch_size = 32
epochs_number = 1

print "Creating the model..."
model = Sequential()
model.add(LSTM(200,  input_shape=(timesteps, dimensions),  return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
fname = './model/lstm2.h5'
model.load_weights(fname)


def predict(text):
    x = np.zeros((1, timesteps, 300), dtype=np.float32)

    tokens = tokenize(text)

    mj = 0

    for w in tokens:
        if(mj < timesteps):

            try:
                x[0][mj] = w2v.word_vec(w)
                mj += 1
            except:
                continue

        else:
            break

    return model.predict(x)


while True:
    input = raw_input("Enter text: ")

    if input == "exit":
        exit()

    prediction = predict(input)[0][0]

    pred_word = "POSITIVE"
    percentage = prediction * 100.0
    if round(prediction) == 0:
        pred_word = "NEGATIVE"
        percentage = 100 - percentage


    print('My prediction is that this text is {}. I am {:.2f}% sure.'.format(pred_word, percentage))

