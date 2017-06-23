# import gensim
# from gensim.utils import tokenize
# from sklearn.datasets import load_files
# import numpy as np
#
# train = load_files("./data/aclImdb/train/")
# x_train, y_train = train.data, train.target
#
# # test = load_files("./data/aclImdb/test/")
# # x_test, y_test = test.data, test.target
#
#
# print('Loading w2v...')
# model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
#
# features = 400
#
# x = np.zeros((25000, features, 300), dtype=np.float32)
# y = np.array(y_train)
#
# i = 0
#
# for index, row in enumerate(x_train):
#
#     tokens = tokenize(row)
#
#     mj = 0
#
#     for w in tokens:
#         if(mj < features):
#
#             try:
#                 x[i][mj] = model.word_vec(w)
#                 mj += 1
#             except:
#                 continue
#
#         else:
#             break
#
#     i += 1
#     print ((i / 25000.0) * 100), "% complete                                                                        \r",
#
#
# del model
#
# # x = np.array(x)
# # y = np.array(y)
#
# np.save("./data/saves/x_train", x)
# np.save("./data/saves/y_train", y)
#
# exit()

# ----------------------------------------------------------------------------------------------------------------------
from keras import callbacks
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
import numpy as np

timesteps = 400
dimensions = 300
batch_size = 32
epochs_number = 17

print "Loading data..."
# train_X = np.load("./data/saves/x_train.npy")
# train_Y = np.load("./data/saves/y_train.npy")
test_X = np.load("./data/saves/x_test.npy")
test_Y = np.load("./data/saves/y_test.npy")

print "Creating the model..."
model = Sequential()
model.add(LSTM(200,  input_shape=(timesteps, dimensions),  return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])

fname = './model/lstm2.h5'

model.load_weights(fname)

cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=3),
            callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)]

# print "Training the model..."
# model.fit(train_X, train_Y,
#           batch_size=batch_size,
#           epochs=epochs_number,
#           callbacks=cbks,
#           validation_data=(test_X, test_Y))

loss, acc = model.evaluate(test_X, test_Y, batch_size)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

print "Exporting the model..."

# ----------------------------------------------------------------------------------------------------------------------

"""
Features: 350 - x5 - best = val_loss: 0.2867 - val_acc: 0.8882 (Epoch 12)

Epoch 1/40
25000/25000 [==============================] - 565s - loss: 0.6896 - acc: 0.5260 - val_loss: 0.6621 - val_acc: 0.6236
Epoch 2/40
25000/25000 [==============================] - 586s - loss: 0.6529 - acc: 0.6314 - val_loss: 0.6306 - val_acc: 0.6693
Epoch 3/40
25000/25000 [==============================] - 628s - loss: 0.6284 - acc: 0.6647 - val_loss: 0.5803 - val_acc: 0.7407
Epoch 4/40
25000/25000 [==============================] - 699s - loss: 0.5942 - acc: 0.6997 - val_loss: 0.6757 - val_acc: 0.5434
Epoch 5/40
25000/25000 [==============================] - 704s - loss: 0.4534 - acc: 0.7762 - val_loss: 0.3189 - val_acc: 0.8684
Epoch 6/40
25000/25000 [==============================] - 710s - loss: 0.3121 - acc: 0.8712 - val_loss: 0.2837 - val_acc: 0.8822
Epoch 7/40
25000/25000 [==============================] - 674s - loss: 0.4112 - acc: 0.7833 - val_loss: 0.2813 - val_acc: 0.8825
Epoch 8/40
25000/25000 [==============================] - 660s - loss: 0.2636 - acc: 0.8928 - val_loss: 0.3358 - val_acc: 0.8575
Epoch 9/40
25000/25000 [==============================] - 696s - loss: 0.2589 - acc: 0.8952 - val_loss: 0.3297 - val_acc: 0.8709
Epoch 10/40
25000/25000 [==============================] - 792s - loss: 0.2300 - acc: 0.9101 - val_loss: 0.3045 - val_acc: 0.8792
Epoch 11/40
25000/25000 [==============================] - 1244s - loss: 0.2022 - acc: 0.9231 - val_loss: 0.2952 - val_acc: 0.8872
Epoch 12/40
25000/25000 [==============================] - 642s - loss: 0.1806 - acc: 0.9321 - val_loss: 0.2867 - val_acc: 0.8882
Epoch 13/40
25000/25000 [==============================] - 612s - loss: 0.1570 - acc: 0.9434 - val_loss: 0.3535 - val_acc: 0.8682
Epoch 14/40
25000/25000 [==============================] - 600s - loss: 0.1310 - acc: 0.9539 - val_loss: 0.3173 - val_acc: 0.8885
Epoch 15/40
25000/25000 [==============================] - 607s - loss: 0.1066 - acc: 0.9640 - val_loss: 0.3428 - val_acc: 0.8847


# ----------------------------------------------------------------------------------------------------------------------

Features: 400 - x6 - best = val_loss: 0.2634 - val_acc: 0.8940 (Epoch 10)

Epoch 1/40
25000/25000 [==============================] - 599s - loss: 0.6899 - acc: 0.5174 - val_loss: 0.6846 - val_acc: 0.5251
Epoch 2/40
25000/25000 [==============================] - 612s - loss: 0.6600 - acc: 0.6091 - val_loss: 0.6145 - val_acc: 0.6980
Epoch 3/40
25000/25000 [==============================] - 623s - loss: 0.6268 - acc: 0.6654 - val_loss: 0.7180 - val_acc: 0.6306
Epoch 4/40
25000/25000 [==============================] - 634s - loss: 0.6113 - acc: 0.6804 - val_loss: 0.5526 - val_acc: 0.7476
Epoch 5/40
25000/25000 [==============================] - 639s - loss: 0.5200 - acc: 0.7620 - val_loss: 0.4150 - val_acc: 0.8274
Epoch 6/40
25000/25000 [==============================] - 643s - loss: 0.3273 - acc: 0.8643 - val_loss: 0.3405 - val_acc: 0.8551
Epoch 7/40
25000/25000 [==============================] - 647s - loss: 0.2867 - acc: 0.8823 - val_loss: 0.3644 - val_acc: 0.8350
Epoch 8/40
25000/25000 [==============================] - 649s - loss: 0.2592 - acc: 0.8972 - val_loss: 0.2643 - val_acc: 0.8910
Epoch 9/40
25000/25000 [==============================] - 655s - loss: 0.2365 - acc: 0.9066 - val_loss: 0.3108 - val_acc: 0.8798
Epoch 10/40
25000/25000 [==============================] - 652s - loss: 0.2108 - acc: 0.9172 - val_loss: 0.2634 - val_acc: 0.8940
Epoch 11/40
25000/25000 [==============================] - 658s - loss: 0.1872 - acc: 0.9302 - val_loss: 0.2884 - val_acc: 0.8889
Epoch 12/40
25000/25000 [==============================] - 656s - loss: 0.1628 - acc: 0.9389 - val_loss: 0.2821 - val_acc: 0.8914
Epoch 13/40
25000/25000 [==============================] - 662s - loss: 0.1362 - acc: 0.9509 - val_loss: 0.4336 - val_acc: 0.8539
Epoch 14/40
25000/25000 [==============================] - 658s - loss: 0.1117 - acc: 0.9617 - val_loss: 0.3068 - val_acc: 0.8876


"""
