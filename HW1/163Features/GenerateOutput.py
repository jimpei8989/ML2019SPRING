import sys
import numpy as np
import pandas as pd

def ReadTrainingData(path):
    df = pd.read_csv(path, encoding="big5", header = None)
    data = df.drop(columns = [0, 1]).values.astype(np.float64)
    num = data.shape[0] // 18
    X = np.concatenate([data[18 * i : 18 * (i + 1), :].reshape((1, -1)) for i in range(num)], axis = 0)
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
