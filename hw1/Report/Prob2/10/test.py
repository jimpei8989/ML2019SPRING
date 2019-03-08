import numpy as np
import pandas as pd

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTestingData(path, mean, stdd):
    df = pd.read_csv(path, encoding="big5", header = None).replace("NR", 0)
    raw_data = df.drop(columns = [0, 1]).values.astype(np.float64)
    num = raw_data.shape[0] // 18
    data = np.concatenate([raw_data[18 * d : 18 * (d+1), :].T for d in range(num)])
    data = (data - mean) / stdd

    X = np.concatenate([data[9 * i + 4 : 9 * (i + 1), 9].reshape((1, -1)) for i in range(num)], axis = 0)
    return np.concatenate([np.ones((num, 1)), X], axis = 1), X.shape[0]

if __name__ == "__main__":
    with np.load("result.npz") as npf:
        w = npf['w']
        mean = npf['mean']
        stdd = npf['stdd']

    X, num = ReadTestingData("../../../data/test.csv", mean, stdd)
    Y = np.dot(X, w.T)

    df = pd.DataFrame(Y, columns = ["value"])
    df.insert(loc = 0, column = "id",value = ["id_"+str(i) for i in range(num)])
    df.to_csv("result.csv", index = False)
