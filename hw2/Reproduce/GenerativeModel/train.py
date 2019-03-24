import sys
import numpy as np
import pandas as pd

def ReadTrainingData(path_X, path_Y):
    rX = pd.read_csv(path_X).drop(["fnlwgt"], axis = 1).values.astype(np.float64)
    rY = pd.read_csv(path_Y).values.astype(np.float64).reshape((-1))

    mean = np.mean(rX, axis = 0).reshape((1, -1))
    stdd = np.std(rX, axis = 0).reshape((1, -1))
    zX = (rX - mean) / stdd
    return zX, rY, mean, stdd

def Accuracy(Y, Ypred):
    return np.mean((Y == Ypred))

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)

    Xtrain_csv = sys.argv[1]
    Ytrain_csv = sys.argv[2]
    data_npz = sys.argv[3]

    X, Y, mean, stdd = ReadTrainingData(Xtrain_csv, Ytrain_csv)
    num, dim = X.shape
    print("Num, Dim = {}".format(X.shape))

    Mean, covMat, Xnum = np.empty((2, X.shape[1])), np.empty((2, X.shape[1], X.shape[1])), np.empty(2)

    for y in [0, 1]:
        Xp = X[Y == y]
        Mean[y] = np.mean(Xp, axis = 0)
        covMat[y] = np.cov(Xp.T)
        Xnum[y] = Xp.shape[0]
        print(Mean[y].shape, covMat[y].shape)

    print(Xnum)

    cov = (covMat[0] * Xnum[0] + covMat[1] * Xnum[1]) / num

    b = -0.5 * np.dot(np.dot(Mean[0], np.linalg.inv(cov)), Mean[0]) + 0.5 * np.dot(np.dot(Mean[1], np.linalg.inv(cov)), Mean[1]) + np.log(Xnum[0] / Xnum[1])
    w = np.dot((Mean[0] - Mean[1]).T, np.linalg.inv(cov))

    print(Accuracy(Y, (np.dot(X, w.T) + b <= 0.5)))
    np.savez(data_npz, mean = mean, stdd = stdd, w = w, b = b)
