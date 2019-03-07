import numpy as np
import pandas as pd

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTrainingData(path, std = False, selected = [l for l in range(18)]):
    df = pd.read_csv(path, encoding="big5").drop(["測站"], axis = 1).replace("NR", 0)
    raw_data = df.drop(columns = ["日期", "測項"]).values.astype(np.float64)
    rdata = np.concatenate([raw_data[18 * d : 18 * (d + 1), :].T for d in range(20 * 12)], axis = 0)

    for m in range(12):
        for h in range(480):
            if rdata[480*m + h, 9] == -1:
                hp = 0
                while hp < 480 and rdata[480*m + hp] is not -1:
                    break
                d = hp - h
                rdata[480*m + h, 9] = 0 if h == 0 else (rdata[480*m + h-1, 9] * (d-1) / d) + (rdata[480*m + hp, 9] if hp != 480 else 1) / d

    mean = np.mean(rdata, axis = 0).reshape((1, -1)) if std is True else np.zeros((1, rdata.shape[1]))
    stdd = np.std(rdata, axis = 0).reshape((1, -1))  if std is True else np.ones((1, rdata.shape[1])) 
    zdata = (rdata - mean) / stdd

    X = np.concatenate([zdata, zdata * zdata], axis = 1)
    Y = rdata[:, 9].reshape((-1, 1))
    return mean, stdd, X, Y

def SelectData(X, Y, selected = [l for l in range(18)], cut = 360):
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
        if epoch % 1000 == 0:
            print("- Epoch: %6d, loss = %f, RMSE(Grad) = %f" % (epoch, Loss(w, X, Y), RMSE(grad)))
    return w

if __name__ == "__main__":
    mean, stdd, X, Y = ReadTrainingData("../data/train.csv", std = True)

    line_cols = [l for l in range(14)]
    quad_cols = [8, 9, 10, 14]
    selected = line_cols + sorted([q + 18 for q in quad_cols])

    trainX, trainY, validX, validY = SelectData(X, Y, selected = selected)
    
    eta = 1e-3
    epochs = 1e5
    lamb = 0

    w = GradientDescent(trainX, trainY, eta = eta, epochs = epochs, lamb = lamb)
    Ein, Evalid = (float(Loss(w, trainX, trainY)), float(Loss(w, validX, validY)))
    print("Ein = %f, Evalid = %f" % (Ein, Evalid))

    np.savez("result.npz", w = w, mean = mean, stdd = stdd)
