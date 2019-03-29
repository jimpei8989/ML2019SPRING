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
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    testCSV = sys.argv[1] if len(sys.argv) == 4 else "../../data/test.csv"
    outputCSV = sys.argv[2] if len(sys.argv) == 4 else "./predict.csv"
    summary = sys.argv[3] if len(sys.argv) == 4 else "./model.summary"
    X, num, Xdim, Ydim = ReadTestingData(testCSV)

    model = load_model("model.h5")

    Y = np.argmax(model.predict(X), axis = 1)

    df = pd.DataFrame(Y, columns = ["label"])
    df.insert(loc = 0, column = "id",value = [i for i in range(num)])
    df.to_csv(outputCSV, index = False)

    with open(summary, "w+") as f:
        print(model.summary, file = f)
