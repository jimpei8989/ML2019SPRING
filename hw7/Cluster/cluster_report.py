import os, sys, time, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.manifold import TSNE

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
        latentDim = 16
        self.Encoder = nn.Sequential(
                nn.Conv2d( 3, 32, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(),
                nn.Conv2d(64, 96, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(),
                )
        self.Transform = nn.Sequential(
                nn.Linear(96 * 4 * 4, latentDim),
                )
        self.Detransform = nn.Sequential(
                nn.Linear(latentDim, 96 * 4 * 4),
                nn.ReLU(),
                )
        self.Decoder = nn.Sequential(
                nn.ConvTranspose2d(96, 64, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32,  3, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
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
    visNpy      = sys.argv[3]

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    plt.switch_backend('agg')
    luckyNumber = int('703747d01aee6f863de4856ce0352f2d2ba164f18a5616d195634f45ef17c729', 16) % (2 ** 32)
    np.random.seed(luckyNumber)
    torch.manual_seed(luckyNumber)

    epochs, batchSize = 50, 128

    print('Load AutoEncoder')
    with open(AEPath, 'rb') as f:
        AE = pickle.load(f)
    AE.cuda().eval()

    # Problem c
    data = np.load(visNpy)
    data = np.transpose(data, (0, 3, 1, 2)).astype('float') / 255
    data = torch.tensor(data).float()
    dataset = TensorDataset(data)

    dataloader = DataLoader(dataset, batch_size = batchSize, shuffle = False, num_workers = 16, pin_memory = True)
    codes = [AE.encode(inputs.cuda()) for (inputs, ) in dataloader]
    codes = torch.cat(codes).detach().cpu().numpy()
    codes = (codes - np.mean(codes, axis = 0)) / np.std(codes, axis = 0)

    tsneTimestamp = time.time()
    print('--- Begin TSNE ---')
    tsne = TSNE(n_components = 2, random_state = luckyNumber)
    transformed = tsne.fit_transform(codes)
    print(transformed.shape)
    print('Elapsed Time: {:2.3f}s'.format(time.time() - tsneTimestamp))
    print('---  End TSNE  ---')

    fig, ax = plt.subplots()
    x, y = transformed[:, 0], transformed[:, 1]
    ax.scatter(x[:2500], y[:2500], color = 'red', s = 2)
    ax.scatter(x[2500:], y[2500:], color = 'blue', s = 2)

    fig.savefig('tsne.png', dpi = 150)

if __name__ == "__main__":
    main()
        
