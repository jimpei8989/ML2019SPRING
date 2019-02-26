import numpy as np
import pandas as pd

def ReadTrainingData(path):
    df = pd.read_csv(path, encoding="big5").drop(["測站"], axis = 1).replace("NR", 0)
    raw_data = df.drop(columns = ["日期", "測項"]).values.astype(np.float64)
    data = np.concatenate([raw_data[18 * d : 18 * (d + 1), :].T for d in range(20 * 12)], axis = 0)

    mean = np.mean(data, axis = 0).reshape((1, -1))
    stdd = np.std(data, axis = 0).reshape((1, -1))

    X = np.concatenate([data[480*m + h : 480*m + h+9, 9].reshape((1, -1)) for h in range(471) for m in range(12)], axis = 0)
    Y = np.concatenate([data[480*m + h+9, 9].reshape((1, -1))             for h in range(471) for m in range(12)], axis = 0)
    num = X.shape[0]

    # Standardize
    data = (data - mean) / stdd
    zX = np.concatenate([data[480*m + h : 480*m + h+9, 9].reshape((1, -1)) for h in range(471) for m in range(12)], axis = 0)

    return np.concatenate([np.ones((num, 1)), X], axis = 1), np.concatenate([np.ones((num, 1)), zX], axis = 1), Y, mean, stdd

def ReadTestingData(path, mean, stdd):
    df = pd.read_csv(path, encoding="big5", header = None).replace("NR", 0)
    raw_data = df.drop(columns = [0, 1]).values.astype(np.float64)
    num = raw_data.shape[0] // 18
    data = np.concatenate([raw_data[18 * d : 18 * (d+1), :].T for d in range(num)])

    X = np.concatenate([data[9 * i : 9 * (i + 1), 9].reshape((1, -1)) for i in range(num)], axis = 0)

    # Standardize
    data = (data - mean) / stdd
    zX = np.concatenate([data[9 * i : 9 * (i + 1), 9].reshape((1, -1)) for i in range(num)], axis = 0)

    return np.concatenate([np.ones((num, 1)), X], axis = 1), np.concatenate([np.ones((num, 1)), zX], axis = 1)

X, zX, Y, mean, stdd = ReadTrainingData("train.csv")
np.savez("train.npz", X = X, Y = Y, mean = mean, stdd = stdd)
np.savez("ztrain.npz", X = zX, Y = Y, mean = mean, stdd = stdd)

X, zX = ReadTestingData("test.csv", mean, stdd)
np.savez("test.npz", X = X)
np.savez("ztest.npz", X = zX)
