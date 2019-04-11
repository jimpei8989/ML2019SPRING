import sys, os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.metrics import confusion_matrix

def ReadTrainingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    print("Num = {}, Xdim = {}, Ydim = {}".format(num, Xdim, Ydim))
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0) / 255
    Y = df['label'].values.reshape((-1, 1))
    return X, Y, X.shape[0]

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    plt.switch_backend('agg')

    trainCSV = sys.argv[1]
    modelH5 = sys.argv[2]
    outputFig = sys.argv[3]

    model = load_model(modelH5)
    X, Y, num = ReadTrainingData(trainCSV)
    X, Y = X[num // 10 * 9 :, :], Y[num // 10 * 9:, :]

    YPredict = np.argmax(model.predict(X), axis = 1).reshape((-1, 1))

    CM = confusion_matrix(Y, YPredict).astype('float')
    CM = CM / CM.sum(axis = 1).reshape((-1, 1))

    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    fig, ax = plt.subplots()
    im = ax.imshow(CM, cmap = 'GnBu')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks = np.arange(CM.shape[1]),
            yticks = np.arange(CM.shape[0]),
            xticklabels = classes,
            yticklabels = classes,
            title = "Confusion Matrix",
            xlabel = 'Predicted label',
            ylabel = 'True label')

    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            ax.text(j, i, "%.2f" % (CM[i, j]),
                    ha = "center", va = "center",
                    color = "white" if CM[i, j] > 0.5 else "black")

    fig.savefig(outputFig, dpi = 150)

