import os, sys, time, pickle
import numpy as np
import pandas as pd
from skimage.io import imread
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.cluster import KMeans

def ReadData(path):
    readTimestamp = time.time()
    print('--- Begin Reading Images ---')
    imgs = np.concatenate([np.expand_dims(np.transpose(imread('{}/{:06d}.jpg'.format(path, i)), axes = (2, 0, 1)), axis = 0) for i in range(1, 40001)], axis = 0) / 255
    print('Elapsed Time: {:2.3f}s'.format(time.time() - readTimestamp))
    print('---  End Reading Images  ---')
    return imgs

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        latentDim = 24
        self.Encoder = nn.Sequential(
                nn.Conv2d( 3, 12, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(),
                nn.Conv2d(12, 24, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(),
                nn.Conv2d(24, 48, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(),
                )
        self.Transform = nn.Sequential(
                nn.Linear(48 * 4 * 4, latentDim),
                )
        self.Detransform = nn.Sequential(
                nn.Linear(latentDim, 48 * 4 * 4),
                nn.ReLU(),
                )
        self.Decoder = nn.Sequential(
                nn.ConvTranspose2d(48, 24, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
                nn.ReLU(),
                nn.ConvTranspose2d(24, 16, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
                nn.ReLU(),
                nn.ConvTranspose2d(16,  3, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
                nn.ReLU(),
                )
    
    def forward(self, x):
        x = self.Encoder(x)
        originalSize = x.size()
        x = x.contiguous().view(x.shape[0], -1)
        x = self.Transform(x)
        x = self.Detransform(x)
        x = x.contiguous().view(originalSize)
        x = self.Decoder(x)
        return x

    def encode(self, x):
        x = self.Encoder(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.Transform(x)
        return x

def main():
    # Misk
    imgPath     = sys.argv[1]
    AEPath      = sys.argv[2]
    testcaseCsv = sys.argv[3]
    outputCsv   = sys.argv[4]

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    luckyNumber = int('703747d01aee6f863de4856ce0352f2d2ba164f18a5616d195634f45ef17c729', 16) % (2 ** 32)
    np.random.seed(luckyNumber)
    torch.manual_seed(luckyNumber)

    epochs, batchSize = 100, 128

    X = torch.tensor(ReadData(imgPath)).float()
    dataset = TensorDataset(X)

    # AutoEncoder
    try:
        with open(AEPath, 'rb') as f:
            AE = pickle.load(f)
        print('Load AutoEncoder')
    except FileNotFoundError:
        # Train
        print('Train AutoEncoder')
        AE = AutoEncoder().cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(AE.parameters())
        trainLoader = DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 16, pin_memory = True)
        AE.train()
        for epoch in range(1, epochs + 1):
            beginTimestamp, losses = time.time(), list()
            for (inputs, ) in trainLoader:
                inputs = inputs.cuda()
                AE.zero_grad()
                outputs = AE(inputs)
                loss = criterion(outputs, inputs)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            meanLoss = np.mean(losses) * batchSize
            print('Epoch: {:3d}/{}'.format(epoch, epochs), 'TrainingLoss: {:1.6f}'.format(meanLoss), 'Elapsed Time: {:2.3f}s'.format(time.time() - beginTimestamp), sep = '\t')
        torch.save(AE.state_dict(), AEPath)

    AE.eval()
    testLoader = DataLoader(dataset, batch_size = batchSize, shuffle = False, num_workers = 16, pin_memory = True)

    codes = [AE.encode(inputs.cuda()) for (inputs, ) in testLoader]
    codes = torch.cat(codes).detach().cpu().numpy()
    codes = (codes - np.mean(codes, axis = 0)) / np.std(codes, axis = 0)

    # K-Means
    kmeansTimestamp = time.time()
    print('--- Begin K-Means ---')
    kmeans = KMeans(n_clusters = 2, n_init = 128, max_iter = 1000, n_jobs = 16, random_state = luckyNumber)
    Y = kmeans.fit_predict(codes)
    print('Elapsed Time: {:2.3f}s'.format(time.time() - kmeansTimestamp))
    print('---  End K-Means  ---')

    # Output
    outputTimestamp = time.time()
    print('--- Begin Output ---')
    testcase = pd.read_csv(testcaseCsv).values[:, 1:] - 1 
    ans = (Y[testcase[:, 0]] == Y[testcase[:, 1]]).astype('int')
    outputDF = pd.DataFrame(zip(range(ans.shape[0]), ans), columns = ['id', 'label'])
    outputDF.to_csv(outputCsv, index = None)
    print('Elapsed Time: {:2.3f}s'.format(time.time() - outputTimestamp))
    print('---  End Output  ---')

if __name__ == "__main__":
    main()
        
