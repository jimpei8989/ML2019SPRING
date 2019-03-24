import sys
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.ensemble import GradientBoostingClassifier

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
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    data_npz = sys.argv[3]
    model_pkl = sys.argv[4]

    with np.load(data_npz) as npf:
        mean = npf['mean']
        stdd = npf['stdd']

    with open(model_pkl, "rb") as f:
        clf = pk.load(f)

    X, num = ReadTestingData(input_csv, mean, stdd, Q = 1)
    Y = clf.predict(X).astype(int)

    df = pd.DataFrame(Y, columns = ["label"])
    df.insert(loc = 0, column = "id",value = [i for i in range(1, num + 1)])
    df.to_csv(output_csv, index = False)
