import sys, os
import numpy as np
import pandas as pd
import pickle 
np.random.seed(50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32))

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

def ReadTrainingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    print("Num = {}, Xdim = {}, Ydim = {}".format(num, Xdim, Ydim))
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0)
    Y = np.zeros((num, Ydim))
    Y[np.arange(num), df['label'].values] = 1
    return X / 255, Y, num, Xdim, Ydim

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    trainCSV = sys.argv[1] if len(sys.argv) == 2 else "../../data/train.csv"
    X, Y, num, Xdim, Ydim = ReadTrainingData(trainCSV)

    datagen = ImageDataGenerator(rotation_range = 20, width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True, validation_split = 0.1)
    datagen.fit(X)

    model = Sequential()
    model.add(Conv2D(filters =  16, kernel_size = (3, 3), input_shape = (48, 48, 1), padding='same'))
    model.add(Conv2D(filters =  32, kernel_size = (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters =  64, kernel_size = (3, 3), padding='same'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='same'))
    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Dropout(0.5))

    model.add(Dense(Ydim, activation='softmax'))

    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
            
    trainGenerator = datagen.flow(X, Y, batch_size = 128, subset = "training")
    validGenerator = datagen.flow(X, Y, batch_size = 128, subset = "validation")
    trainNum, validNum = len(trainGenerator), len(validGenerator)

    model.fit_generator( trainGenerator, steps_per_epoch = trainNum, validation_data = validGenerator, validation_steps = validNum, epochs = 300)

    model.save("model.h5")

    score = model.evaluate(X, Y, verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
