import sys
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def ReadTrainingData(path_X, path_Y, Q = 2):
    squaredTerms = ["age", "capital_gain", "capital_loss", "hours_per_week"]
    dfX = pd.read_csv(path_X).drop(["fnlwgt"], axis = 1)
    for q in range(2, Q + 1):
        for t in squaredTerms:
            dfX[t + "^%d" % (q)] = dfX[t] ** q

    dfY = pd.read_csv(path_Y)

    rX = dfX.values.astype(np.float64)
    rY = dfY.values.astype(np.float64).reshape((-1, 1))

    data = np.concatenate([rX, rY], axis = 1)
    np.random.shuffle(data)
    rX, rY = data[:, :-1], data[:, -1].reshape((-1, 1))

    mean = np.mean(rX, axis = 0).reshape((1, -1))
    stdd = np.std(rX, axis = 0).reshape((1, -1))
    zX = (rX - mean) / stdd
    return np.concatenate([np.ones((zX.shape[0], 1)), zX], axis = 1), rY.reshape(-1), mean, stdd


if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    Xtrain_csv = "../../data/X_train.csv"
    Ytrain_csv = "../../data/Y_train.csv"

    X, Y, mean, stdd = ReadTrainingData(Xtrain_csv, Ytrain_csv, Q = 1)

    etas = [10 ** -(k / 4) for k in range(3, 6)]
    depths = [4, 5, 6]
    
    for eta in etas:
        for d in depths:
            clf = GradientBoostingClassifier(n_estimators = 127, learning_rate = eta, max_depth = d, random_state = lucky_num)
            scores = cross_val_score(clf, X, Y, cv = 5, scoring = 'accuracy', n_jobs = -1)
            print("- eta = %f, depth = %d, Mean = %.8f, Min = %.8f" % (eta, d, np.mean(scores), np.min(scores)))
