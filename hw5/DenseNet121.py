import sys, os
import numpy as np
from skimage.io import imread, imsave

import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd.gradcheck import zero_gradients
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 

def DeprocessImg(x, Mean, Std):
    return np.clip((np.transpose(np.squeeze(x, axis = 0), (1, 2, 0)) * Std + Mean) * 255, 0, 255).astype('uint8')

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    inputDir = sys.argv[1]
    outputDir = sys.argv[2]

    Mean = np.array([0.485, 0.456, 0.406])
    Std = np.array([0.229, 0.224, 0.225])
    model = densenet121(pretrained = True).cuda()
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    trans = transform.Compose([transform.ToTensor(), transform.Normalize(mean = Mean, std = Std)])

    epsilon = 23 / 255 / 0.225

    for idx in range(200):
        Input = imread("%s/%03d.png" % (inputDir, idx))
        InputTensor = trans(Input).unsqueeze(0).cuda()
        InputTensor.requires_grad = True
        zero_gradients(InputTensor)

        OutputTensor = model(InputTensor)
        OutputLabel  = np.argmax(model(InputTensor).cpu().detach().numpy())

        loss = criterion(OutputTensor, torch.from_numpy(np.array([OutputLabel])).cuda())
        loss.backward()

        Output = DeprocessImg((InputTensor + epsilon * InputTensor.grad.sign_()).detach().cpu().numpy(), Mean, Std)
        imsave("%s/%03d.png" % (outputDir, idx), Output)

        print('-> %3d' % idx, end = '\r' if idx != 199 else '\n')

