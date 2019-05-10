import sys, os, jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from keras.callbacks import ReduceLROnPlateau

def Segment(filename):
    df = pd.read_csv(filename)
    return [[s for s in sentence.split(' ')] for sentence in ' '.join(jieba.cut('\n'.join(df['comment']))).split('\n')]

def S2V(s, wordMap):
    ret = np.zeros(len(wordMap) + 1)
    for w in s:
        try:
            ret[wordMap[w]] += 1
        except KeyError:
            ret[0] += 1
    return ret.reshape((1, -1))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    plt.switch_backend('agg')

    dictTxt         = sys.argv[1]
    trainXPath      = sys.argv[2]
    trainYPath      = sys.argv[3]
    testXPath       = sys.argv[4]
    predictPath     = sys.argv[5]
    w2vModelPath    = sys.argv[6]
    modelPath       = sys.argv[7]
    figPath         = sys.argv[8]

    jieba.load_userdict(dictTxt)
    jieba.enable_parallel()

    print("--- Begin Reading Data ---")
    trainComments = Segment(trainXPath)
    testComments = Segment(testXPath)
    df = pd.read_csv(trainYPath)
    trainLabels = df['label'].values.reshape((-1, 1))

    wordCount, wordMap = {}, {}
    for sentence in trainComments + testComments:
        for word in sentence:
            try:
                wordCount[word] += 1
            except KeyError:
                wordCount[word] = 1
    for word in wordCount:
        if wordCount[word] >= 10:
            wordMap[word] = len(wordMap) + 1

    print("> Word Map has size = %d" % (len(wordMap)))
    print("--- End Reading Data ---")

    dim = len(wordMap)
    trainX = np.concatenate([S2V(sentence, wordMap) for sentence in trainComments], axis = 0)
    trainY = trainLabels.reshape((-1, 1))
    print(trainX.shape, trainY.shape)

    model = Sequential()
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'relu'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

    history = model.fit(trainX, trainY, batch_size = 480, epochs = 3, validation_split = 0.2)
    model.save(modelPath)

    testX = np.concatenate([S2V(sentence, wordMap) for sentence in testComments], axis = 0)
    predictY = model.predict(testX, batch_size = 480)

    outputDF = pd.DataFrame(zip(range(testX.shape[0]), (predictY.reshape(-1) > 0.5).astype(int)), columns = ['id', 'label'])
    outputDF.to_csv(predictPath, index = None)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize = (8, 4))

    ax[0].plot(history.history['acc'], color = '#1e90ff', label = 'Training Accuracy')
    ax[0].plot(history.history['val_acc'], color = '#32cd32', label = 'Testing Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(history.history['loss'], color = '#1e90ff', label = 'Training Loss')
    ax[1].plot(history.history['val_loss'], color = '#32cd32', label = 'Testing Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    fig.savefig(figPath, dpi = 150)

