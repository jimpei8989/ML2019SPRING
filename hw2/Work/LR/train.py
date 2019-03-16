import sys
import numpy as np
import pandas as pd

def ReadTrainingData(path_X, path_Y, Q = 2):
    squaredTerms = ["age", "capital_gain", "capital_loss", "hours_per_week"]
    dfX = pd.read_csv(path_X).drop(["fnlwgt"], axis = 1)
    for q in range(2, Q + 1):
        for t in squaredTerms:
            dfX[t + "^%d" % (q)] = dfX[t] ** q
    dfY = pd.read_csv(path_Y)

    rX = dfX.values.astype(np.float64)
    rY = dfY.values.astype(np.float64).reshape((-1, 1))

    data = np.concatenate([rX, rY], axis = 1)
    np.random.shuffle(data)

    rX, rY = data[:, :-1], data[:, -1].reshape((-1, 1))

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

def GradientDescent(X, Y, eta = 1e-3, epochs = 5e3, batchsize = 128):
    num, dim = X.shape
    w = np.zeros((1, dim))

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m, v = np.float64(0), np.float64(0)

    for epoch in range(1, int(epochs) + 1):
        for idx in range(num // batchsize):
            bX, bY = X[idx * batchsize : (idx + 1) * batchsize, :], Y[idx * batchsize : (idx + 1) * batchsize, :]
            grad = np.sum(-(bY - Sigmoid(np.dot(bX, w.T))) * bX, axis = 0).reshape((1, -1))

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            mhat = (m / (1 - (beta1 ** epoch))).reshape((1, -1))
            vhat = (v / (1 - (beta2 ** epoch))).reshape((1, -1))
            w -= eta / (np.sqrt(vhat) + eps) * mhat

        if epoch % 10 == 0:
            print("- Epoch: %6d, loss = %f, RMSE(Grad) = %f" % (epoch, Loss(w, X, Y), RMSE(grad)))
    return w

if __name__ == "__main__":
    Xtrain_csv = "../../data/X_train.csv"
    Ytrain_csv = "../../data/Y_train.csv"
    np_file = "result.npz"

    X, Y, mean, stdd = ReadTrainingData(Xtrain_csv, Ytrain_csv, Q = 3)
    print("Num, Dim = {}".format(X.shape))
    w = GradientDescent(X, Y)
    np.savez(np_file, w = w, mean = mean, stdd = stdd)

