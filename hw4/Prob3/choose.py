import sys, os
import numpy as np
import pandas as pd
import pickle

from keras.models import load_model

def ReadTrainingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    print("Num = {}, Xdim = {}, Ydim = {}".format(num, Xdim, Ydim))
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0)
    Y = df['label'].values.reshape((-1, 1))
    return X / 255, Y, num, Xdim, Ydim

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)

    modelH5 = sys.argv[1]
    trainCSV = sys.argv[2]

    model = load_model(modelH5)
    X, Y, num, Xdim, Ydim = ReadTrainingData(trainCSV)

    predictY = model.predict(X)
    chosenIndices = []

    for label in range(Ydim):
        candidates = [idx for idx in range(num) if Y[idx] == np.argmax(predictY[idx])]
        chosen = sorted([(idx, predictY[idx, label]) for idx in candidates], key = lambda k: k[1], reverse = True)[31][0]
        #  chosen = np.random.choice(candidates)
        chosenIndices.append(chosen)
        print(label, chosen)

    print(chosenIndices)

