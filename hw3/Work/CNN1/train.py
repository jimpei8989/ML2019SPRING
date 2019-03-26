import sys
import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def ReadTrainingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    print("Num = {}, Xdim = {}, Ydim = {}".format(num, Xdim, Ydim))
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0)
    Y = np.zeros((num, Ydim))
    Y[np.arange(num), df['label'].values] = 1
    return X, Y, num, Xdim, Ydim

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)

    trainCSV = sys.argv[1] if len(sys.argv) == 2 else "../../data/train.csv"
    X, Y, num, Xdim, Ydim = ReadTrainingData(trainCSV)

    model = Sequential()
    model.add(Conv2D(filters =  64, kernel_size = (4, 4), strides = (2, 2), input_shape = (48, 48, 1)))
    model.add(MaxPooling2D(pool_size = (3, 3)))
    #  model.add(Conv2D(filters = 128, kernel_size = (4, 4), strides = (2, 2)))
    #  model.add(MaxPooling2D(pool_size = (3, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    #  model.add(Dense(128, activation="relu")
    model.add(Dense(Ydim, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X, Y, batch_size = 128, epochs = 20, verbose = 2, validation_split = 0.1)

    score = model.evaluate(X, Y, verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
