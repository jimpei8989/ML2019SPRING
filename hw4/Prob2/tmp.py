import os, sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from tensorflow import set_random_seed

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

    model = load_model(modelH5)
    layerDict = dict([(layer.name, layer) for layer in model.layers])

    layerName = "leaky_re_lu_2"

    layerOutput = layerDict[layerName].output

    filterIndex = 26

    loss = K.mean(layerOutput[:, :, :, filterIndex])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])

    inputImgData = (np.random.random((1, 48, 48, 1)) * 20 + 118) / 255

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    eta, epochs = 1e-3, 100
    m, v = 0, 0

    for epoch in range(1, epochs + 1):
        lossValue, gradsValue = iterate([inputImgData])
        m = beta1 * m + (1 - beta1) * gradsValue
        v = beta2 * v + (1 - beta2) * (gradsValue ** 2)
        mhat = m / (1 - (beta1 ** epoch))
        vhat = v / (1 - (beta2 ** epoch))
        inputImgData += eta / (np.sqrt(vhat) + eps) * mhat

    plt.imsave("OMG.png", DeprocessImg(np.squeeze(inputImgData, axis = (0, 3))), cmap = "BuPu")

