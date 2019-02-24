import time
import numpy as np
import pandas as pd

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTrainingData(path):
    df = pd.read_csv(path, encoding="big5").drop(["測站"], axis = 1)
    raw_data = df.drop(columns = ["日期", "測項"]).values.astype(np.float64)
    data = np.concatenate([raw_data[18 * d : 18 * (d + 1), :].T for d in range(20 * 12)], axis = 0)
    X = np.concatenate([data[480*m + h : 480*m + h+9, :].reshape((1, -1)) for h in range(471) for m in range(12)], axis = 0)
    Y = np.concatenate([data[480*m + h+9, 9].reshape((1, -1))             for h in range(471) for m in range(12)], axis = 0)
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1), Y

def RMSE(x):
    return (np.dot(x.reshape((1, -1)), x.reshape((-1, 1))) / x.reshape((1, -1)).shape[0]) ** 0.5

def Loss(w, X, Y):
    return (np.sum((np.dot(X, w.T) - Y) * (np.dot(X, w.T) - Y)) / X.shape[0]) ** 0.5

def Grad(w, X, Y):
    return np.sum(2 * (Y - np.dot(X, w.T)) * (-X), axis = 0) / X.shape[0]

X, Y = ReadTrainingData("../data/data.csv")

num, dim = X.shape

eta = 1e-6
iterationTimes = int(1e4)

w = np.zeros((1, dim))

for epoch in range(iterationTimes):
    w -= eta * Grad(w, X, Y)
    if epoch % 1000 == 0:
        print("- Epoch: %6d, loss = %f, RMSE(Grad) = %f" % (epoch, Loss(w, X, Y), RMSE(Grad(w, X, Y))))

print(w)
np.save("result_%s.npy" % (time.strftime("%m-%d_%H-%M", time.gmtime())), w)
