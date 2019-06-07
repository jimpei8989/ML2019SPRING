import os, sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torchvision import transforms

# torchsummary
from torchsummary import summary

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

                    ConvDW(64, 64, 1),
                    ConvDW(64, 64, 1),
                    ConvDW(64, 128, 2),      # 24 * 24

                    ConvDW(128, 128, 1),
                    ConvDW(128, 128, 1),
                    ConvDW(128, 192, 2),      # 12 * 12 

                    ConvDW(192, 192, 1),
                    ConvDW(192, 192, 2),      # 6 * 6
                    nn.Dropout(0.3),

                    nn.AvgPool2d(3),        # 2 * 2
                )

        self.fc = nn.Sequential(
                    nn.Linear(2 * 2 * 192, 7),
                    nn.Softmax(dim = 1)
                )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

def LoadTrainingData(path):
    print('--- Begin Loading Training Data ---')
    loadTS = time.time()
    df = pd.read_csv(path)
    X = np.concatenate([np.array([float(e) for e in x.split(' ')]).reshape((1, 1, 48, 48)) for x in df['feature']], axis = 0) / 255
    Y = df['label'].values.reshape((-1, 1))
    print('--- End Loading Training Data (Elapsed Time: {:2.3f}) ---'.format(time.time() - loadTS))
    return X, Y.astype('float'), X.shape[0]

def loadTestingData(path):
    print('--- Begin Loading Testing Data ---')
    loadTS = time.time()
    df = pd.read_csv(path)
    X = np.concatenate([np.array([float(e) for e in x.split(' ')]).reshape((1, 1, 48, 48)) for x in df['feature']], axis = 0) / 255
    print('--- End Loading Testing Data (Elapsed Time: {:2.3f}) ---'.format(time.time() - loadTS))
    return X, X.shape[0]


def main():
    trainCSV    = sys.argv[1]
    modelPath   = sys.argv[2]
    testCSV     = sys.argv[3]
    predictCSV  = sys.argv[4]

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    luckyNumber = int('703747d01aee6f863de4856ce0352f2d2ba164f18a5616d195634f45ef17c729', 16) % (2 ** 32)
    np.random.seed(luckyNumber)
    torch.manual_seed(luckyNumber)

    model = MobileNet().cuda()
    try:
        # Load Model
        model.load_state_dict(torch.load(modelPath))
        print('> Load Model')
    except FileNotFoundError:
        # Train Model
        print('> Train Model')

        summary(model, (1, 48, 48))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.25, patience = 7, threshold = 5e-3, cooldown = 7, min_lr = 1e-5, verbose = True)
        epochs, batchSize = 120, 128

        X, Y, num = LoadTrainingData(trainCSV)
        Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X, Y, test_size = 0.2, random_state = luckyNumber)
        trainNum, validNum = Xtrain.shape[0], Xvalid.shape[0]

        trainSet = TensorDataset(torch.Tensor(Xtrain), torch.LongTensor(Ytrain))
        validSet = TensorDataset(torch.Tensor(Xvalid), torch.LongTensor(Yvalid))
        trainLoader = DataLoader(trainSet, batch_size = batchSize, shuffle = True, num_workers = 16, pin_memory = True)
        validLoader = DataLoader(validSet, batch_size = batchSize, shuffle = False, num_workers = 16, pin_memory = True)

        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomAffine(degrees = 10, translate = (0.1, 0.1), scale = (0.9, 1.05)),
                transforms.ToTensor(),
                ])

        for epoch in range(1, epochs + 1):
            beginTS, trainAccu, trainLoss, validAccu, validLoss = time.time(), 0, 0, 0, 0

            model.train()
            for (ipt, opt, ) in trainLoader:
                optimizer.zero_grad()
                ipt = torch.cat([transform(x.view((1, 48, 48))).view(1, 1, 48, 48) for x in (ipt * 255).int()], dim = 0).float() / 255
                pred = model(ipt.cuda())
                loss = criterion(pred, opt.squeeze().cuda())
                loss.backward()
                optimizer.step()

                trainAccu += np.sum(np.argmax(pred.cpu().data.numpy(), axis = 1) == opt.squeeze().numpy()) / trainNum
                trainLoss += loss.item() * ipt.size()[0] / trainNum

            model.eval()
            for (ipt, opt, ) in validLoader:
                pred = model(ipt.cuda())
                loss = criterion(pred, opt.squeeze().cuda())

                validAccu += np.sum(np.argmax(pred.cpu().data.numpy(), axis = 1) == opt.squeeze().numpy()) / validNum
                validLoss += loss.item() * ipt.size()[0] / validNum

            scheduler.step(trainLoss)

            print('Epoch: {:3d}/{}'.format(epoch, epochs), 'Elapsed Time: {:2.3f}s'.format(time.time() - beginTS), sep = '\t')
            print(' -->\tTrainLoss: {:1.6f}, TrainAccu: {:1.6f}\tValidLoss: {:1.6f}, ValidAccu: {:1.6f}'.format(trainLoss, trainAccu, validLoss, validAccu))

        model.half()
        torch.save(model.state_dict(), modelPath)

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

