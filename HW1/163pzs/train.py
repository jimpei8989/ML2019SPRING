import numpy as np
import pandas as pd

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTrainingData(path, std = False):
    df = pd.read_csv(path, encoding="big5").drop(["測站"], axis = 1).replace("NR", 0)
    raw_data = df.drop(columns = ["日期", "測項"]).values.astype(np.float64)

    # rdata is num * dim array
    rdata = np.concatenate([raw_data[18 * d : 18 * (d + 1), :].T for d in range(20 * 12)], axis = 0)

    for m in range(12):
        for h in range(480):
            if rdata[480*m + h, 9] == -1:
                rdata[480*m + h, 9] = 0 if h == 0 else rdata[480*m + h-1, 9]


    mean = np.mean(rdata, axis = 0).reshape((1, -1)) if std is True else np.zeros((1, rdata.shape[1]))
    stdd = np.std(rdata, axis = 0).reshape((1, -1))  if std is True else np.ones((1, rdata.shape[1])) 
    zdata = (rdata - mean) / stdd

    cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
    tmpX = np.concatenate([zdata[:, col].reshape((-1, 1)) for col in cols], axis = 1)
    # zdata = np.concatenate([zdata, tmpX * tmpX], axis = 1)

    X = np.concatenate([zdata[480*m + h : 480*m + h+9, :].reshape((1, -1)) for h in range(471) for m in range(12)], axis = 0)
    Y = np.concatenate([rdata[480*m + h+9, 9].reshape((1, -1))             for h in range(471) for m in range(12)], axis = 0)

    return np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1), Y, mean, stdd

def RMSE(x):
    return (np.dot(x.reshape((1, -1)), x.reshape((-1, 1))) / x.reshape((1, -1)).shape[0]) ** 0.5

def Loss(w, X, Y):
    return (np.sum((np.dot(X, w.T) - Y) * (np.dot(X, w.T) - Y)) / X.shape[0]) ** 0.5

def GradientDescent(X, Y, eta, epochs, lamb = 0):
    num, dim = X.shape
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)

    w = np.zeros((1, dim))
    beta1, beta2, eps = 0.9, 0.9999, 1e-8
    m, v = np.float64(0), np.float64(0)

    for epoch in range(1, int(epochs) + 1):
        grad = np.dot(XTX, w.T) - XTY + 2 * lamb / num * w.T
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = (m / (1 - (beta1 ** epoch))).reshape((1, -1))
        vhat = (v / (1 - (beta2 ** epoch))).reshape((1, -1))
        w -= eta / (np.sqrt(vhat) + eps) * mhat
        if epoch % 1000 == 0:
            print("- Epoch: %6d, loss = %f, RMSE(Grad) = %f" % (epoch, Loss(w, X, Y), RMSE(grad)))
    return w

if __name__ == "__main__":
    trainX, trainY, mean, stdd = ReadTrainingData("../data/train.csv", std = True)

    eta = 1e-3
    epochs = 1e5
    lamb = 1e5

    w = GradientDescent(trainX, trainY, eta = eta, epochs = epochs, lamb = lamb)
    print(w)
    
    np.savez("result.npz", w = w, mean = mean, stdd = stdd)
