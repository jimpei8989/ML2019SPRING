import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as K
from tensorflow import set_random_seed

def ReadTrainingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0) / 255
    Y = df['label'].values.reshape(-1).astype(int)
    return np.concatenate([X, X, X], axis = 3), to_categorical(Y) 

def DeprocessImg(x):
    return np.clip(np.clip((x - np.mean(x)) / (np.std(x) + 1e-5) * 0.1 + 0.5, 0, 1) * 255, 0, 255).astype('uint8')

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)
    set_random_seed(lucky_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    plt.switch_backend('agg')

    modelH5 = sys.argv[1]
    trainCSV = sys.argv[2]
    outputDir = sys.argv[3]

    model = load_model(modelH5)
    X, Y = ReadTrainingData(trainCSV)

    chosenIndices = [9956, 6488, 8805, 82, 16245, 392, 672]
    for idx, chosenIndex in enumerate(chosenIndices):
        fig, ax = plt.subplots(1, 3, figsize=(8, 4))
        x, y = np.expand_dims(X[idx], axis = 0), np.expand_dims(Y[idx], axis = 0)

        loss = K.categorical_crossentropy(y, model.output)
        grads = K.gradients(loss, model.input)[0]
        func = K.function([model.input], [grads])

        print(func([x]))
        saliency = func([x])

        m, s = np.mean(saliency), np.std(saliency)

        ax[0].imshow(x)
        ax[0].set_title("Original")

        im = ax[1].imshow(saliency, cmap = "jet")
        ax[1].set_title("SaliencyMap")
        fig.colorbar(im, ax = ax[1])

        im = ax[2].imshow(np.clip(saliency, m - 2 * s, m + 2 * s), cmap = "binary")
        ax[2].set_title("Saliency within $2 * \sigma$")
        fig.colorbar(im, ax = ax[2])

        fig.savefig("%s/fig1_%d.jpeg" % (outputDir, idx), dpi = 300)

