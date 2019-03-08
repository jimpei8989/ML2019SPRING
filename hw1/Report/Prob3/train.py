import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTrainingData(path, std = False):
    df = pd.read_csv(path, encoding="big5").drop(["測站"], axis = 1).replace("NR", 0)
    raw_data = df.drop(columns = ["日期", "測項"]).values.astype(np.float64)
    rdata = np.concatenate([raw_data[18 * d : 18 * (d + 1), :].T for d in range(20 * 12)], axis = 0)

    mean = np.mean(rdata, axis = 0).reshape((1, -1)) if std is True else np.zeros((1, rdata.shape[1]))
    stdd = np.std(rdata, axis = 0).reshape((1, -1))  if std is True else np.ones((1, rdata.shape[1])) 
    zdata = (rdata - mean) / stdd

    t163X = np.concatenate([zdata[480*m + h : 480*m + h+9, :].reshape((1, -1)) for h in range(471) for m in range(12)], axis = 0)
    t163Y = np.concatenate([rdata[480*m + h+9, 9].reshape((1, -1))             for h in range(471) for m in range(12)], axis = 0)
    t9X = np.concatenate([zdata[480*m + h : 480*m + h+9, 9].reshape((1, -1)) for h in range(471) for m in range(12)], axis = 0)
    t9Y = np.concatenate([rdata[480*m + h+9, 9].reshape((1, -1))             for h in range(471) for m in range(12)], axis = 0)
    return np.concatenate([np.ones((t163X.shape[0], 1)), t163X], axis = 1), t163Y, np.concatenate([np.ones((t9X.shape[0], 1)), t9X], axis = 1), t9Y, mean, stdd

def RMSE(x):
    return (np.dot(x.reshape((1, -1)), x.reshape((-1, 1))) / x.reshape((1, -1)).shape[0]) ** 0.5

def Loss(w, X, Y):
    return (np.sum((np.dot(X, w.T) - Y) * (np.dot(X, w.T) - Y)) / X.shape[0]) ** 0.5

def GradientDescent(X, Y, eta, epochs, lamb):
    num, dim = X.shape
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)
    EinHistory = []

    w = np.zeros((1, dim))
    sigma = np.float64(0)

    for epoch in range(int(epochs)):
        grad = 2 * (np.dot(XTX, w.T) - XTY + lamb * w.T)
        sigma += np.dot(grad.T, grad)
        w -= eta * grad.reshape((1, -1)) / np.sqrt(sigma)
        if (epoch + 1) % 10 == 0:
            EinHistory.append(Loss(w, X, Y))
    return w, EinHistory

if __name__ == "__main__":
    train163X, train163Y, train9X, train9Y, mean, stdd = ReadTrainingData("../../data/train.csv", True)
    np.savez("result.npz", mean = mean, stdd = stdd)

    eta = 1e0
    epochs = 1e5
    lambs = [0.1, 0.01, 0.001, 0.0001]

    EinHistories = dict()

    for lamb in lambs:
        print("Training lambda = %f" % (lamb))
        try:
            with np.load("results/result_%.4f_1.npz" % (lamb)) as npf1, np.load("results/result_%.4f_2.npz" % (lamb)) as npf2:
                w_1 = npf1['w']
                EinHistory_1 = npf1['EinHis']
                w_2 = npf2['w']
                EinHistory_2 = npf2['EinHis']
        except FileNotFoundError:
            w, EinHistory_1 = GradientDescent(train163X, train163Y, eta = eta, epochs = epochs, lamb = lamb)
            np.savez("results/result_%.4f_1.npz" % (lamb), w = w, EinHis = EinHistory_1)
            w, EinHistory_2 = GradientDescent(train9X, train9Y, eta = eta, epochs = epochs, lamb = lamb)
            np.savez("results/result_%.4f_2.npz" % (lamb), w = w, EinHis = EinHistory_2)

        EinHistories[(lamb, 1)] = EinHistory_1
        EinHistories[(lamb, 2)] = EinHistory_2

    t = np.array([10 * tt for tt in range(int(epochs) // 10)])
    colors = ["#48C9B0", "#F7DC6F", "#FFA07A", "#F08080"]
    for i, lamb in enumerate(lambs):
        plt.plot(t, EinHistories[(lamb, 1)], color = colors[i], label = "Model 1, $\lambda = %f$" % (lamb))
        plt.plot(t, EinHistories[(lamb, 2)], color = colors[i], linestyle = "dashed", label = "Model 2, $\lambda = %f$" % (lamb))
        print("- Model 1, lambda = %.4f, Ein = %.8f" % (lamb, EinHistories[(lamb, 1)][-1]))
        print("- Model 2, lambda = %.4f, Ein = %.8f" % (lamb, EinHistories[(lamb, 2)][-1]))

    plt.title("RMSE(Training Data) : epochs")
    plt.xlabel("epoch")
    plt.ylabel("Error (RMSE)")
    plt.legend()
    plt.savefig("result.png", dpi = 300)
