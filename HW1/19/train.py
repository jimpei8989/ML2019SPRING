import numpy as np
import pandas as pd

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTrainingData(path, std = False):
    df = pd.read_csv(path, encoding="big5").drop(["測站"], axis = 1).replace("NR", 0)
    raw_data = df.drop(columns = ["日期", "測項"]).values.astype(np.float64)
    data = np.concatenate([raw_data[18 * d : 18 * (d + 1), :].T for d in range(20 * 12)], axis = 0)

    mean = np.mean(data, axis = 0).reshape((1, -1)) if std is True else np.zeros((1, data.shape[1]))
    stdd = np.std(data, axis = 0).reshape((1, -1))  if std is True else np.ones((1, data.shape[1])) 
    data = (data - mean) / stdd

    X = np.concatenate([data[480*m + h : 480*m + h+9, 8 : 10].reshape((1, -1)) for h in range(471) for m in range(12)], axis = 0)
    Y = np.concatenate([data[480*m + h+9, 9].reshape((1, -1))             for h in range(471) for m in range(12)], axis = 0)
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1), Y, mean, stdd

def RMSE(x):
    return (np.dot(x.reshape((1, -1)), x.reshape((-1, 1))) / x.reshape((1, -1)).shape[0]) ** 0.5

def Loss(w, X, Y):
    return (np.sum((np.dot(X, w.T) - Y) * (np.dot(X, w.T) - Y)) / X.shape[0]) ** 0.5

def GradientDescent(X, Y, eta, epochs):
    num, dim = X.shape
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)

    w = np.zeros((1, dim))
    sigma = np.float64(0)

    for epoch in range(int(epochs)):
        grad = np.dot(XTX, w.T) - XTY
        sigma += np.dot(grad.T, grad)
        w -= eta * grad.reshape((1, -1)) / np.sqrt(sigma)
        if (epoch + 1) % 1000 == 0:
            print("- Epoch: %6d, loss = %f, RMSE(Grad) = %f" % (epoch, Loss(w, X, Y), RMSE(grad)))
    return w

if __name__ == "__main__":
    trainX, trainY, mean, stdd = ReadTrainingData("../data/train.csv")

    eta = 1e3
    epochs = 7e5

    w = GradientDescent(trainX, trainY, eta = eta, epochs = epochs)
    np.savez("result.npz", w = w, mean = mean, stdd = stdd)
