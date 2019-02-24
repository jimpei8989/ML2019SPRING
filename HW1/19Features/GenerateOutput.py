import sys
import numpy as np
import pandas as pd

def ReadTrainingData(path):
    df = pd.read_csv(path, encoding="big5", header = None)
    raw_data = df.drop(columns = [0, 1]).values.astype(np.float64)
    num = raw_data.shape[0] // 18
    data = np.concatenate([raw_data[18 * d : 18 * (d+1), :].T for d in range(num)])
    X = np.concatenate([data[9 * i : 9 * (i + 1), 8:10].reshape((1, -1)) for i in range(num)], axis = 0)
    return np.concatenate([np.ones((num, 1)), X], axis = 1)

if (len(sys.argv) != 2):
    print("usage:\tpython3 GenerateOutput NPY_FILE", file=sys.stderr)
    exit(1)

name = sys.argv[1][:-4]

w = np.load(name + ".npy")
X = ReadTrainingData("../data/test_1.csv")
Y = np.dot(X, w.T)

df = pd.DataFrame(Y, columns = ["value"])
df.insert(loc = 0, column = "id",value = ["id_" + str(i) for i in range(X.shape[0])])
df.to_csv(name + ".csv", index = False)
