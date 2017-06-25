from keras import callbacks
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
import numpy as np

timesteps = 400
dimensions = 300
batch_size = 32
epochs_number = 17

print "Loading data..."
train_X = np.load("./data/saves/x_train.npy")
train_Y = np.load("./data/saves/y_train.npy")
test_X = np.load("./data/saves/x_test.npy")
test_Y = np.load("./data/saves/y_test.npy")

print "Creating the model..."
model = Sequential()
model.add(LSTM(200,  input_shape=(timesteps, dimensions),  return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])

fname = './model/lstm2.h5'

# model.load_weights(fname)

cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=3),
            callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)]

print "Training the model..."
model.fit(train_X, train_Y,
          batch_size=batch_size,
          epochs=epochs_number,
          callbacks=cbks,
          validation_data=(test_X, test_Y))

loss, acc = model.evaluate(test_X, test_Y, batch_size)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

print "Exporting the model..."