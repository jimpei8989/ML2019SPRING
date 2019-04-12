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

    #  os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #  plt.switch_backend('agg')

    modelH5 = sys.argv[1]
    trainCSV = sys.argv[2]
    outputDir = sys.argv[3]

    X, Y, num, Xdim, Ydim = ReadTrainingData(trainCSV)

    model = load_model(modelH5)
    layerDict = dict([(layer.name, layer) for layer in model.layers])

    layerNames = ["conv2d_2", "conv2d_4"]
    filterNumbers = [32, 48]
    chosenInput = X[5607].reshape((1, 48, 48, 1))

    fig, ax = plt.subplots(8, 10, figsize = (16, 16))
    fig.suptitle("%s (4 on the left)  and %s (6 on the right)" % (layerNames[0], layerNames[1]), fontsize = 28)

    for idx, (layerName, filterNumber) in enumerate(zip(layerNames, filterNumbers)):
        CalOutput = K.function([model.input, K.learning_phase()], [layerDict[layerName].output])
        outputImg = CalOutput([chosenInput, 0])[0]

        for filterIndex in range(filterNumber):
            r, c = (filterIndex // 4, filterIndex % 4) if idx == 0 else (filterIndex // 6, filterIndex % 6 + 4)
            ax[r, c].imshow(DeprocessImg(outputImg[0, :, :, filterIndex]), cmap = "BuPu", interpolation='none')
            ax[r, c].set_title("Filter {}".format(filterIndex))
            ax[r, c].axis('off')
    fig.savefig("%s/fig2_2.jpg" % (outputDir), dpi = 150)

