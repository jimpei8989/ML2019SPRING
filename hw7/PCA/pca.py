import os, sys, time
import numpy as np 
from skimage.io import imread, imsave

def ReadData(path):
    readTimestamp = time.time()
    fileList = os.listdir(path)

    print('--- Begin Reading Images ---')
    imgShape = imread(os.path.join(path, fileList[0])).shape
    imgs = np.concatenate([imread(os.path.join(path, f)).reshape((1, -1)) for f in fileList], axis = 0).astype('float')
    print('Elapsed Time: {:2.3f}s'.format(time.time() - readTimestamp))
    print('---  End Reading Images  ---')
    return imgs, imgShape

def Process(M, originalShape):
    M -= np.min(M)
    M /= np.max(M)
    return (M * 255).astype('uint8').reshape(originalShape)

def main():
    path = '../data/Aberdeen'

    imgs, originalShape = ReadData(path)
    mean = np.mean(imgs, axis = 0)
    imgs -= mean

    # Problem a
    imsave('average.jpg', Process(mean, originalShape))

    # Problem b
    svdTimestamp = time.time()
    print('--- Begin SVD ---')
    u, s, v = np.linalg.svd(imgs, full_matrices = False)
    print('Elapsed Time: {:2.3f}s'.format(time.time() - svdTimestamp))
    print('---  End SVD  ---')

    for i in range(5):
        eigenface = Process(v[i], originalShape)
        imsave(str(i) + '_eigenface.jpg', eigenface.reshape(originalShape)) 

    # Problem c
    K = 5
    reconstruct = (u[:, :K] * s[:K]) @ v[:K, :] + mean
    #  chosenIndices = [1, 10, 22, 37, 72]
    chosenIndices = np.random.choice(imgs.shape[0], size = 5)
    
    for i, idx in enumerate(chosenIndices): 
        output = Process(reconstruct[idx], originalShape)
        imsave(str(i) + '_reconstruction.jpg', output)

    # Problem d
    for i in range(5):
        print(s[i] * 100 / np.sum(s))

if __name__ == "__main__":
    main()

