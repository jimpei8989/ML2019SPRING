import os, sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torchvision import transforms

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        def ConvBN(inp, oup, stride):
            return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias = False),
                    nn.BatchNorm2d(oup),
                    nn.LeakyReLU(negative_slope = 0.1, inplace=True),
                )

        def ConvDW(inp, oup, stride):
            return nn.Sequential(
		    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias = False),
		    nn.BatchNorm2d(inp),
		    nn.LeakyReLU(negative_slope = 0.1, inplace=True),

		    nn.Conv2d(inp, oup, 1, 1, 0, bias = False),
		    nn.BatchNorm2d(oup),
		    nn.LeakyReLU(negative_slope = 0.1, inplace=True),
		)

        self.conv = nn.Sequential(
                    ConvBN(1, 64, 1),
                    #  nn.Dropout(0.2),

                    ConvDW(64, 64, 1),
                    ConvDW(64, 64, 1),
                    ConvDW(64, 96, 2),      # 24 * 24
                    #  nn.Dropout(0.3),

                    ConvDW(96, 96, 1),
                    ConvDW(96, 128, 2),      # 12 * 12 
                    nn.Dropout(0.3),

                    ConvDW(128, 128, 1),
                    ConvDW(128, 128, 2),      # 6 * 6
                    nn.Dropout(0.3),

                    nn.AvgPool2d(3),        # 2 * 2
                )

        self.fc = nn.Sequential(
                    nn.Linear(2 * 2 * 128, 7),
                    nn.Softmax(dim = 1)
                )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

def loadTestingData(path):
    print('--- Begin Loading Testing Data ---')
    loadTS = time.time()
    df = pd.read_csv(path)
    X = np.concatenate([np.array([float(e) for e in x.split(' ')]).reshape((1, 1, 48, 48)) for x in df['feature']], axis = 0) / 255
    print('--- End Loading Testing Data (Elapsed Time: {:2.3f}) ---'.format(time.time() - loadTS))
    return X, X.shape[0]


def main():
    modelPath   = sys.argv[1]
    testCSV     = sys.argv[2]
    predictCSV  = sys.argv[3]

    #  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    luckyNumber = int('703747d01aee6f863de4856ce0352f2d2ba164f18a5616d195634f45ef17c729', 16) % (2 ** 32)
    np.random.seed(luckyNumber)
    torch.manual_seed(luckyNumber)

    model = MobileNet().cuda()

    # Load Model
    model.load_state_dict(torch.load(modelPath))
    print('> Load Model')

    # Predict
    model.eval()
    model.float()
    Xtest, testNum = loadTestingData(testCSV)

    predictTS = time.time()
    print('--- Begin Predicting ---')
    testSet = TensorDataset(torch.Tensor(Xtest))
    testLoader = DataLoader(testSet, batch_size = 256, shuffle = False, num_workers = 16, pin_memory = True)
    scoreY = np.concatenate([model(ipt.cuda()).cpu().detach().numpy() for (ipt, ) in testLoader], axis = 0)
    predictY = np.argmax(scoreY, axis = 1)

    df = pd.DataFrame(data = {'id' : [_ for _ in range(testNum)], 'label' : predictY.reshape(-1)})
    df.to_csv(predictCSV, index = False)

    print('--- End Predicting (Elapsed Time: {:2.3f}s)'.format(time.time() - predictTS))

if __name__ == "__main__":
    main()

