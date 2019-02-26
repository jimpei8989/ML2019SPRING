import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def RMSE(x):
    return (np.dot(x.reshape((1, -1)), x.reshape((-1, 1))) / x.reshape((1, -1)).shape[0]) ** 0.5

def Loss(w, X, Y):
    return (np.sum((np.dot(X, w.T) - Y) * (np.dot(X, w.T) - Y)) / X.shape[0]) ** 0.5

def GradientDescent(X, Y, eta = 1e-6, epochs = 1e5):
    num, dim = X.shape
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)

    w = np.zeros((1, dim))
    sigma = np.float64(0)

    for epoch in range(int(epochs)):
        grad = np.dot(XTX, w.T) - XTY
        sigma += np.dot(grad.T, grad)
        w -= eta * grad.reshape((1, -1)) / np.sqrt(sigma)
        if epoch % 100 == 0:
            print("- Epoch: %6d, loss = %f, RMSE(Grad) = %f" % (epoch, Loss(w, X, Y), RMSE(grad)))

    return w


if __name__ == "__main__":
    with np.load("data/train.npz") as npfile:
        trainX, trainY = npfile['X'], npfile['Y']

    w = GradientDescent(trainX, trainY, eta = 1e-0, epochs = 1e5)

    with np.load("data/test.npz") as npfile:
        testX = npfile['X']
        predict_Y = np.dot(testX, w.T)
        testN = testX.shape[0]

    df = pd.DataFrame(predict_Y, columns = ["values"])
    df.insert(loc = 0, column = "id",value = ["id_"+str(i) for i in range(testN)])
    print(df)
    df.to_csv("results/result.csv", index = False)

