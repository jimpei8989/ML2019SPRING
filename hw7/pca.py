import os, sys, time
import numpy as np 
from skimage.io import imread, imsave

def ReadData(path):
    readTimestamp = time.time()
    print('--- Begin Reading Images ---')
    fileList = os.listdir(path)
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
    path = sys.argv[1]
    inputPath = sys.argv[2]
    outputPath = sys.argv[3]

    imgs, originalShape = ReadData(path)
    mean = np.mean(imgs, axis = 0)
    imgs -= mean

    svdTimestamp = time.time()
    print('--- Begin SVD ---')
    u, s, v = np.linalg.svd(imgs, full_matrices = False)
    print('Elapsed Time: {:2.3f}s'.format(time.time() - svdTimestamp))
    print('---  End SVD  ---')

    K = 5
    inputImg = imread(inputPath).astype('float').reshape(-1) - mean
    outputImg = Process(np.dot(v[:K].T, np.dot(v[:K], inputImg)) + mean, originalShape)
    imsave(outputPath, outputImg)

if __name__ == "__main__":
    main()

