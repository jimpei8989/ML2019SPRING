import sys, os
import numpy as np
import pandas as pd
import pickle

from keras.models import load_model

def ReadTestingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0)
    return X / 255, num, Xdim, Ydim

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    testCSV = sys.argv[1]
    outputCSV = sys.argv[2]
    X, num, Xdim, Ydim = ReadTestingData(testCSV)

    modelPaths = ["ImageGenNN" + str(i) + "/model.h5" for i in [2, 3, 4, 5]]
    print("Using model:", modelPaths)

    models = [load_model(m) for m in modelPaths]

    prediction = np.squeeze(np.sum([np.expand_dims(model.predict(X), axis = 0) for model in models], axis = 0), axis = 0)
    Y = np.argmax(prediction, axis = 1)

    df = pd.DataFrame(Y, columns = ["label"])
    df.insert(loc = 0, column = "id",value = [i for i in range(num)])
    df.to_csv(outputCSV, index = False)

