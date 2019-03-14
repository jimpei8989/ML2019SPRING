import sys
import numpy as np
import pandas as pd

def ReadTrainingData(path_X, path_Y):
    rX = pd.read_csv(path_X).drop(["fnlwgt"], axis = 1).values.astype(np.float64)
    rY = pd.read_csv(path_Y).values.astype(np.float64).reshape((-1, 1))

    mean = np.mean(rX, axis = 0).reshape((1, -1))
    stdd = np.std(rX, axis = 0).reshape((1, -1))
    zX = (rX - mean) / stdd
    return np.concatenate([np.ones((zX.shape[0], 1)), zX], axis = 1), rY, mean, stdd

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Loss(w, X, Y):
    return np.mean((Sigmoid(np.dot(X, w.T)) > 0.5) == Y)

def RMSE(z):
    return (np.dot(z, z.T) / z.shape[0]) ** 0.5

def GradientDescent(X, Y, eta = 1e-3, epochs = 1e4):
    num, dim = X.shape
    w = np.zeros((1, dim))

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m, v = np.float64(0), np.float64(0)

    for epoch in range(1, int(epochs) + 1):
        grad = np.sum(-(Y - Sigmoid(np.dot(X, w.T))) * X, axis = 0).reshape((1, -1))

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = (m / (1 - (beta1 ** epoch))).reshape((1, -1))
        vhat = (v / (1 - (beta2 ** epoch))).reshape((1, -1))
        w -= eta / (np.sqrt(vhat) + eps) * mhat
        if epoch % 1000 == 0:
            print("- Epoch: %6d, loss = %f, RMSE(Grad) = %f" % (epoch, Loss(w, X, Y), RMSE(grad)))
    return w

if __name__ == "__main__":
    Xtrain_csv = "../../data/X_train.csv"
    Ytrain_csv = "../../data/Y_train.csv"
    np_file = "result.npz"

    X, Y, mean, stdd = ReadTrainingData(Xtrain_csv, Ytrain_csv)
    print("Num, Dim = {}".format(X.shape))
    w = GradientDescent(X, Y)
    np.savez(np_file, w = w, mean = mean, stdd = stdd)

