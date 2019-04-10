import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from tensorflow import set_random_seed

def ReadTrainingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    print("Num = {}, Xdim = {}, Ydim = {}".format(num, Xdim, Ydim))
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0)
    Y = np.zeros((num, Ydim))
    Y[np.arange(num), df['label'].values] = 1
    return X / 255, Y, num, Xdim, Ydim

def DeprocessImg(x):
    return np.clip(np.clip((x - np.mean(x)) / (np.std(x) + 1e-5) * 0.1 + 0.5, 0, 1) * 255, 0, 255).astype('uint8')

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)
    set_random_seed(lucky_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # To make the plt work on Meow
    plt.switch_backend('agg')

    modelH5 = sys.argv[1]
    trainCSV = sys.argv[2]
    outputDir = sys.argv[3]

    X, Y, num, Xdim, Ydim = ReadTrainingData(trainCSV)

    model = load_model(modelH5)
    layerDict = dict([(layer.name, layer) for layer in model.layers])

    layerNames = ["conv2d_2", "conv2d_4"]
    filterNumbers = [32, 48]
    chosenInput = X[5607].reshape((1, 48, 48, 1))

    for idx, (layerName, filterNumber) in enumerate(zip(layerNames, filterNumbers)):
        #  plt.figure(idx)
        fig, ax = plt.subplots(filterNumber // 8, 8, figsize = (12, filterNumber // 4))
        fig.suptitle("Layer: {}".format(layerName), fontsize = 28)

        iterate = K.function([model.input, K.learning_phase()], [layerDict[layerName].output])
        outputImg = iterate([chosenInput, 0])[0]

        for filterIndex in range(filterNumber):
            ax[filterIndex // 8, filterIndex % 8].imshow(DeprocessImg(outputImg[0, :, :, filterIndex]), cmap = "BuPu", interpolation='none')
            ax[filterIndex // 8, filterIndex % 8].set_title("Filter {}".format(filterIndex))
            ax[filterIndex // 8, filterIndex % 8].axis('off')
        plt.savefig(outputDir + '/' + layerName + '.png', dpi = 150)

