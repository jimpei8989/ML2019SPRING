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

def RMSE(z):
    return (np.dot(z, z.T) / z.shape[0]) ** 0.5

def CrossEntropy(w, X, Y):
    def C(w, x, y):
        return np.log(np.dot(x, w.T)) if y[0] == 1 else np.log(1 - np.dot(x, w.T))
    return -np.sum(C(w, X, Y))

def Accuracy(w, X, Y):
    return np.mean((Sigmoid(np.dot(X, w.T)) > 0.5) == Y)

def Predict(W, X):
    return np.sum(Sigmoid(np.dot(X, W.T)), axis = 1).reshape((-1, 1))

def LOSS(W, X, Y):
    return np.mean((Predict(W, X) > 0.5) == Y)

def GradientDescent(X, Y, eta = 1e-4, epochs = 5e3, batchsize = 128):
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

        if epoch % 100 == 0:
            print("\t- Epoch: %6d, CrossEntropy = %f, RMSE(Grad) = %f" % (epoch, CrossEntropy(w, X, Y), RMSE(grad)))
    return w

if __name__ == "__main__":
    Xtrain_csv = "../../data/X_train.csv"
    Ytrain_csv = "../../data/Y_train.csv"
    np_file = "result.npz"

    X, Y, mean, stdd = ReadTrainingData(Xtrain_csv, Ytrain_csv, Q = 4)

    num, dim = X.shape[0] * 3 // 4, X.shape[1]
    
    trainX, trainY = X[:num, :], Y[:num, :]
    validX, validY = X[num:, :], Y[num:, :]

    W = np.empty((0, dim), np.float64)

    print("Num, Dim = {}".format(X.shape))

    for t in range(20):
        print("\n" + "> The %d time boosting" % (t))

        PredY = Predict(W, trainX)
        tY = trainY if t == 0 else trainY - PredY

        w = GradientDescent(trainX, tY)
        W = np.append(W, w, axis = 0)
        print("> End of %d time boosting, Etrain = %f, Evalid = %f" % (t, LOSS(W, trainX, trainY), LOSS(W, validX, validY)))

    np.savez(np_file, W = W, mean = mean, stdd = stdd)
