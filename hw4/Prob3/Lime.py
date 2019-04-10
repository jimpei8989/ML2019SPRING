import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow import set_random_seed
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic

def ReadTrainingData(path):
    df = pd.read_csv(path)
    num, Xdim, Ydim = df.shape[0], 48 * 48, 7
    print("Num = {}, Xdim = {}, Ydim = {}".format(num, Xdim, Ydim))
    X = np.concatenate([np.array([float(f) for f in x.split()]).reshape((1, 48, 48, 1)) for x in df['feature']], axis = 0) / 255
    Y = df['label'].values.reshape(-1).astype(int)
    return np.concatenate([X, X, X], axis = 3), Y

def Lime(model, x, y, seed):
    def Predict(Input):
        return model.predict(np.mean(Input.reshape((-1, 48, 48, 3)), axis = 3).reshape(-1, 48, 48, 1)).reshape((-1, 7))

    def Segmentation(Input):
        return slic(Input)

    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(image=x, classifier_fn=Predict, segmentation_fn=Segmentation, random_seed = seed)
    image, mask = explanation.get_image_and_mask(label=y, positive_only=False, hide_rest=False, num_features=5, min_weight=0.0) 
    return image

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)
    set_random_seed(lucky_num)

    modelH5 = sys.argv[1]
    trainCSV = sys.argv[2]
    outputDir = sys.argv[3]

    model = load_model(modelH5)
    X, Y = ReadTrainingData(trainCSV)

    chosenIndices = [9956, 6488, 8805, 82, 16245, 392, 672]

    for label, idx in enumerate(chosenIndices):
        image = Lime(model, X[idx], label, lucky_num)
        plt.imsave(outputDir + '/fig3_{}.png'.format(label), image)

