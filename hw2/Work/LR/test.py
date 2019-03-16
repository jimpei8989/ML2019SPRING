import sys
import numpy as np
import pandas as pd

from train import Predict

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTestingData(path, mean, stdd, Q = 2):
    squaredTerms = ["age", "capital_gain", "capital_loss", "hours_per_week"]
    dfX = pd.read_csv(path).drop(["fnlwgt"], axis = 1)
    for q in range(2, Q + 1):
        for t in squaredTerms:
            dfX[t + "^%d" % (q)] = dfX[t] ** q

    rX = dfX.values.astype(np.float64)

    zX = (rX - mean) / stdd
    return np.concatenate([np.ones((zX.shape[0], 1)), zX], axis = 1), zX.shape[0]

if __name__ == "__main__":
    np_file = "result.npz"
    input_csv = "../../data/X_test.csv"
    output_csv = "result.csv"

    with np.load(np_file) as npf:
        W = npf['W']
        mean = npf['mean']
        stdd = npf['stdd']

    X, num = ReadTestingData(input_csv, mean, stdd, Q = 4)
    Y = (Predict(W, X) > 0.5).astype(int)

    df = pd.DataFrame(Y, columns = ["label"])
    df.insert(loc = 0, column = "id",value = [i for i in range(1, num + 1)])
    df.to_csv(output_csv, index = False)
