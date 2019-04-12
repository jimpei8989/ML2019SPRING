import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow import set_random_seed

from vis.visualization import visualize_cam

def ReadTrainingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    print("Num = {}, Xdim = {}, Ydim = {}".format(num, Xdim, Ydim))
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0)
    Y = np.zeros((num, Ydim))
    Y[np.arange(num), df['label'].values] = 1
    return X / 255, Y, num, Xdim, Ydim

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)
    set_random_seed(lucky_num)

    #  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #  plt.switch_backend('agg')

    modelH5 = sys.argv[1]
    trainCSV = sys.argv[2]
    outputDir = sys.argv[3]

    model = load_model(modelH5)
    X, Y, num, Xdim, Ydim = ReadTrainingData(trainCSV)
    
    layerDict = dict([(layer.name, idx) for idx, layer in enumerate(model.layers)])
    #  layerDict = dict(layer.name : idx for idx, layer in enumerate(model.layers))
    layerName = "dense_5"
    layerIdx = layerDict[layerName]

    chosenIndices = [17185, 6452, 14028, 4191, 1405, 302, 23051]

    for label, idx in enumerate(chosenIndices):
        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        cam = visualize_cam(model, layerIdx, [label], np.expand_dims(X[idx], axis = 0), penultimate_layer_idx = 4)

        ax[0].imshow(np.squeeze(X[idx], axis = 2), cmap = "gray")
        ax[0].axis('off')
        ax[0].set_title("Original")

        im = ax[1].imshow(cam, cmap = "YlOrRd")
        ax[1].axis('off')
        ax[1].set_title("Class Activation Map")
        fig.colorbar(im, ax = ax[1])

        fig.savefig("%s/fig4_%d.png" % (outputDir, label), dpi = 150)

