import sys, os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from cv2 import GaussianBlur

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    plt.switch_backend('agg')
    plt.subplots_adjust(wspace = 0.3)

    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    
    for idx in range(200):
        Image = imread("%s/%03d.png" % (inputDir, idx))
        imsave("%s/%03d.png" % (outputDir, idx), GaussianBlur(Image, (5, 5), 0))
        print("-> %03d" % (idx), end = '\r')

