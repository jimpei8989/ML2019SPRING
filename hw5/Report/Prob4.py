import sys, os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    np.random.seed(lucky_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    plt.switch_backend('agg')
    plt.subplots_adjust(wspace = 0.3)

    chosenIndices = np.random.choice(200, size = 3)

    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    
    # Label names are found here:
    # http://files.fast.ai/models/imagenet_class_index.json
    with open('../label.pkl', 'rb') as f:
        Dict = pkl.load(f)

    Mean = np.array([0.485, 0.456, 0.406])
    Std = np.array([0.229, 0.224, 0.225])
    model = resnet50(pretrained = True)
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    trans = transform.Compose([transform.ToTensor(), transform.Normalize(mean = Mean, std = Std)])

    for idx in chosenIndices:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        Image = imread("%s/%03d.png" % (inputDir, idx))
        OriPredict = model(trans(imread("%s/%03d.png" % (inputDir, idx))).unsqueeze(0)).detach().numpy()
        AttPredict = model(trans(imread("%s/%03d.png" % (outputDir, idx))).unsqueeze(0)).detach().numpy()

        X = [0, 1, 2]
        OriSorted = sorted([(i, a) for i, a in enumerate(OriPredict.reshape(-1))], key = lambda k : (k[1], k[0]), reverse = True)[:3]
        AttSorted = sorted([(i, a) for i, a in enumerate(AttPredict.reshape(-1))], key = lambda k : (k[1], k[0]), reverse = True)[:3]

        ax[0].axis('off')
        ax[0].imshow(Image)
        ax[0].set_title('Image %3d' % (idx))

        ax[1].bar(X, [pair[1] for pair in OriSorted], width = 0.4, tick_label = [Dict[str(pair[0])][1] for pair in OriSorted], color = '#00bfff')
        ax[1].set_title('Original Prediction')

        ax[2].bar(X, [pair[1] for pair in AttSorted], width = 0.4, tick_label = [Dict[str(pair[0])][1] for pair in AttSorted], color = '#ff69b4')
        ax[2].set_title('Attacked Prediction')

        fig.savefig('Result_%03d.png' % (idx), dpi = 150)

