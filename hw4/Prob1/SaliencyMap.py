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
    print("Num = {}, Xdim = {}, Ydim = {}".format(num, Xdim, Ydim))
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0) / 255
    Y = np.zeros((num, Ydim))
    Y[np.arange(num), df['label'].values] = 1
    return X, Y

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)
    set_random_seed(lucky_num)

    #  os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #  plt.switch_backend('agg')
    plt.subplots_adjust(wspace = 1.0)

    modelH5 = sys.argv[1]
    trainCSV = sys.argv[2]
    outputDir = sys.argv[3]

    model = load_model(modelH5)
    X, Y = ReadTrainingData(trainCSV)

    chosenIndices = [19833, 9612, 24422, 28684, 21428, 2072, 23216]
    for idx, chosenIndex in enumerate(chosenIndices):
        fig, ax = plt.subplots(1, 3, figsize=(8, 4))
        x, y = X[chosenIndex], Y[chosenIndex]

        loss = K.categorical_crossentropy(np.expand_dims(y, axis = 0), model.output)
        grads = K.gradients(loss, model.input)[0]
        CalGrad = K.function([model.input], [grads])

        saliency = np.squeeze(np.abs(CalGrad([np.expand_dims(x, axis = 0)])[0]), axis = 0)
        m, s = np.mean(saliency), np.std(saliency)

        ax[0].imshow(np.squeeze(x, axis = 2), cmap = "gray")
        ax[0].axis('off')
        ax[0].set_title("Original")

        im = ax[1].imshow(np.squeeze(saliency, axis = 2), cmap = "jet")
        ax[1].axis('off')
        ax[1].set_title("SaliencyMap")
        fig.colorbar(im, ax = ax[1])

        im = ax[2].imshow(np.squeeze(np.clip(saliency, m - 2 * s, None), axis = 2), cmap = "afmhot")
        ax[2].axis('off')
        ax[2].set_title("Saliency within $2 * \sigma$")
        fig.colorbar(im, ax = ax[2])

        fig.savefig("%s/fig1_%d.jpg" % (outputDir, idx), dpi = 150)

