import sys
import numpy as np
import pandas as pd

def ReadTrainingData(path):
    df = pd.read_csv(path, encoding="big5", header = None)
    df = df.loc[df[1] == "PM2.5"].drop(columns = [0, 1])
    X = df.values.astype(float)
    num, dim = X.shape
    return np.concatenate([np.ones((num, 1)), X], axis = 1)

w = np.load("result.npy")
X = ReadTrainingData("../data/test.csv")
Y = np.dot(X, w.T)

df = pd.DataFrame(Y, columns = ["value"])
df.insert(loc = 0, column = "id",value = ["id_" + str(i) for i in range(X.shape[0])])
df.to_csv("result.csv", index = False)

