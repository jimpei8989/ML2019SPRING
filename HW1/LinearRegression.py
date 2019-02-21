# Linear Regression with only PM2.5 history

import numpy as np
import pandas as pd

def ReadTrainingData(path):
    df = pd.read_csv(path, encoding="big5").drop(["測站"], axis = 1)
    df = df.loc[df["測項"] == "PM2.5"]
    data = df.drop(columns = ["日期", "測項"]).values.astype(float).reshape(-1)

    num = data.shape[0] - 9
    X = np.concatenate([data[i : i + 9]       for i in range(num)], axis = 0).reshape((num, 9))
    Y = np.concatenate([data[i + 9 : i + 10]  for i in range(num)], axis = 0).reshape((num, 1))

    X = np.concatenate([np.ones((num, 1)), X], axis = 1)

    return X, Y

def Grad(w, X, Y):
    return np.sum(2 * (Y - np.dot(X, w.T)) * (-X), axis = 0)

X, Y = ReadTrainingData("data/train.csv")

num, dim = X.shape

eta = 1e-7
iterationTimes = int(1e5) 

w = np.zeros((1, dim))

for epoch in range(iterationTimes):
    if epoch % 1000 == 0:
        print("--- Epoch: %5d ---" % epoch)
    w -= eta * Grad(w, X, Y)

print(w)
np.save("123.npy", w)
