import gensim
from gensim.utils import tokenize
from sklearn.datasets import load_files
import numpy as np

train = load_files("./data/aclImdb/train/")
x_train, y_train = train.data, train.target

# test = load_files("./data/aclImdb/test/")
# x_test, y_test = test.data, test.target


print('Loading w2v...')
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

features = 400

x = np.zeros((25000, features, 300), dtype=np.float32)
y = np.array(y_train)

i = 0

for index, row in enumerate(x_train):

    tokens = tokenize(row)

    mj = 0

    for w in tokens:
        if(mj < features):

            try:
                x[i][mj] = model.word_vec(w)
                mj += 1
            except:
                continue

        else:
            break

    i += 1
    print ((i / 25000.0) * 100), "% complete                                                                        \r",

del model

np.save("./data/saves/x_train", x)
np.save("./data/saves/y_train", y)

exit()