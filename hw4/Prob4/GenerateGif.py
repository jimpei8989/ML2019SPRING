import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow import set_random_seed

from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)
    set_random_seed(lucky_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    modelH5 = sys.argv[1]
    #  outputDir = sys.argv[2]

    model = load_model(modelH5)
    layerDict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layerName = "dense_5"

    for idx in range(7):
        outputClass = [0]
        losses = [(ActivationMaximization(layerDict[layerName], outputClass), 2),
                    (LPNorm(model.input), 10),
                    (TotalVariation(model.input), 10)]
        opt = Optimizer(model.input, losses)
        opt.minimize(max_iter=500, verbose=True, input_modifiers=[Jitter()], callbacks=[GifGenerator('OptProgress_%d' % (idx))])


