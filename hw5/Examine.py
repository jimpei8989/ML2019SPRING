import sys, os
import numpy as np
from skimage.io import imread, imsave

import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd.gradcheck import zero_gradients
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 

def Predict(model, img):
    return np.argmax(model(img).detach().cpu().numpy())

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    inputDir = sys.argv[1]
    outputDir = sys.argv[2]

    Mean = np.array([0.485, 0.456, 0.406])
    Std = np.array([0.229, 0.224, 0.225])
    model = resnet50(pretrained = True).cuda()
    model.eval()

    trans = transform.Compose([transform.ToTensor(), transform.Normalize(mean = Mean, std = Std)])

    LinfSum = 0
    epsilon = np.pi / 255 / 0.225
    targetClass = np.random.randint(low = 0, high = 1000, size = 1)

    Labels = []
    Linf = []

    for idx in range(200):
        ori = imread("%s/%03d.png" % (inputDir, idx))
        new = imread("%s/%03d.png" % (outputDir, idx))

        oriLabel, newLabel = Predict(model, trans(ori).cuda().unsqueeze(0)), Predict(model, trans(new).cuda().unsqueeze(0))

        Labels.append((oriLabel, newLabel))
        Linf.append(np.max(np.abs(ori.astype('int') - new.astype('int'))))
        print('-> %3d' % idx, end = '\r' if idx != 199 else '\n')

    print('Linf:\t{}'.format(np.mean(Linf)))
    print('Rate:\t{}'.format(np.mean([1 if a != b else 0 for a, b in Labels])))

