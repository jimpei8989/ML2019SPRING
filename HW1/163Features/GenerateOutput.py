import sys
import numpy as np
import pandas as pd

def ReadTrainingData(path, mean, stdd):
    df = pd.read_csv(path, encoding="big5", header = None)
    raw_data = df.drop(columns = [0, 1]).values.astype(np.float64)
    num = raw_data.shape[0] // 18

    data = np.concatenate([raw_data[18 * d : 18 * (d+1), :].T for d in range(num)])
    data = (data - mean) / stdd 

    X = np.concatenate([data[9 * i : 9 * (i + 1), :].reshape((1, -1)) for i in range(num)], axis = 0)
    return np.concatenate([np.ones((num, 1)), X], axis = 1)

if (len(sys.argv) != 2):
    print("usage:\tpython3 GenerateOutput NPY_FILE", file=sys.stderr)
    exit(1)

name = sys.argv[1][:-4]

npfile = np.load(name + ".npz")

w, mean, stdd = npfile['w'], npfile['mean'], npfile['stdd']
X = ReadTrainingData("../data/test_1.csv", mean, stdd)
Y = np.dot(X, w.T) * stdd[0,9] + mean[0,9]

df = pd.DataFrame(Y, columns = ["value"])
df.insert(loc = 0, column = "id",value = ["id_" + str(i) for i in range(X.shape[0])])
df.to_csv(name + ".csv", index = False)
