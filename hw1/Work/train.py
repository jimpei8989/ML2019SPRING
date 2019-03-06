from itertools import product
import numpy as np
import pandas as pd

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTrainingData(path, std = False, cut = 360, selected = [l for l in range(18)]):
    df = pd.read_csv(path, encoding="big5").drop(["測站"], axis = 1).replace("NR", 0)
    raw_data = df.drop(columns = ["日期", "測項"]).values.astype(np.float64)
    rdata = np.concatenate([raw_data[18 * d : 18 * (d + 1), :].T for d in range(20 * 12)], axis = 0)

    for m in range(12):
        for h in range(480):
            if rdata[480*m + h, 9] == -1:
                rdata[480*m + h, 9] = 0 if h == 0 else rdata[480*m + h-1, 9]

    mean = np.mean(rdata, axis = 0).reshape((1, -1)) if std is True else np.zeros((1, rdata.shape[1]))
    stdd = np.std(rdata, axis = 0).reshape((1, -1))  if std is True else np.ones((1, rdata.shape[1])) 
    zdata = (rdata - mean) / stdd

    X = np.concatenate([zdata, zdata * zdata], axis = 1)
    Y = rdata[:, 9].reshape((-1, 1))
    return mean, stdd, X, Y

def SelectData(X, Y, selected = [l for l in range(18)], cut = 240):
    X1 = np.concatenate([X[480*m + h: 480*m + h+9, :].reshape((1, -1)) for h in range(0, cut-9) for m in range(12)], axis = 0)
    Y1 = np.concatenate([Y[480*m + h + 9, :] for h in range(0, cut-9)  for m in range(12)], axis = 0).reshape((-1, 1))
    X2 = np.concatenate([X[480*m + h: 480*m + h+9, :].reshape((1, -1)) for h in range(cut, 471) for m in range(12)], axis = 0)
    Y2 = np.concatenate([Y[480*m + h + 9, :] for h in range(cut, 471)  for m in range(12)], axis = 0).reshape((-1, 1))
    return np.concatenate([np.ones((X1.shape[0], 1)), X1], axis = 1), Y1, np.concatenate([np.ones((X2.shape[0], 1)), X2], axis = 1), Y2

def RMSE(x):
    return (np.dot(x.reshape((1, -1)), x.reshape((-1, 1))) / x.reshape((1, -1)).shape[0]) ** 0.5

def Loss(w, X, Y):
    return (np.dot((np.dot(X, w.T) - Y).T, (np.dot(X, w.T) - Y)) / X.shape[0]) ** 0.5

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
        #  if epoch % 1000 == 0:
        #      print("- Epoch: %6d, loss = %f, RMSE(Grad) = %f" % (epoch, Loss(w, X, Y), RMSE(grad)))
    return w

if __name__ == "__main__":
    mean, stdd, X, Y = ReadTrainingData("../data/train.csv", std = True)

    names = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5", "RAINFALL", "RH", "SO2", "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR"]
    COLUMNS = ["lambda", "Ein1", "Eout1", "Ein2", "Eout2"] + [n + "^1" for n in names] + [n + "^2" for n in names]
    record = pd.DataFrame(columns = COLUMNS)
    
    _ = 0
    for possible in product([0, 1], repeat = 8):
        line_cols = [l for l in range(18)]
        quad_cols = [8, 9, 10, 13] + [i + 1 for i in range(7) if possible[i] is 1] + [i + 5 for i in range(7, 8) if possible[i] is 1]
        selected = line_cols + sorted([q + 18 for q in quad_cols])

        X1, Y1, X2, Y2 = SelectData(X, Y, selected = selected)
        
        for i in range(4, 12):
            eta = 1e-3
            epochs = 1e5
            lamb = 10 ** (i / 2)

            w1 = GradientDescent(X1, Y1, eta = eta, epochs = epochs, lamb = lamb)
            Ein1, Eout1 = (float(Loss(w1, X1, Y1)), float(Loss(w1, X2, Y2)))
            w2 = GradientDescent(X2, Y2, eta = eta, epochs = epochs, lamb = lamb)
            Ein2, Eout2 = (float(Loss(w2, X2, Y2)), float(Loss(w2, X1, Y1)))
            record = record.append(pd.Series([lamb, Ein1, Eout1, Ein2, Eout2] + [1 if l in line_cols else 0 for l in range(18)] + [1 if q in quad_cols else 0 for q in range(18)], index = COLUMNS), ignore_index = True)

        _ += 1
        print("--- %d / 512" % _)
        record.to_csv("record.csv")
