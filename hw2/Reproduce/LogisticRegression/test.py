import sys
import numpy as np
import pandas as pd

#AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR

def ReadTestingData(path, mean, stdd):
    X = pd.read_csv(path).drop(["fnlwgt"], axis = 1).values.astype(np.float64)
    zX = (X - mean) / stdd
    return np.concatenate([np.ones((zX.shape[0], 1)), zX], axis = 1), X.shape[0]

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    data_npz = sys.argv[3]

    with np.load(data_npz) as npf:
        w = npf['w']
        mean = npf['mean']
        stdd = npf['stdd']

    X, num = ReadTestingData(input_csv, mean, stdd)
    Y = (Sigmoid(np.dot(X, w.T)) > 0.5).astype(int)

    df = pd.DataFrame(Y, columns = ["label"])
    df.insert(loc = 0, column = "id",value = [i for i in range(1, num + 1)])
    df.to_csv(output_csv, index = False)
