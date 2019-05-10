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
    return [[word for word in sentence] for sentence in df['comment']]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    plt.switch_backend('agg')
    plt.subplots_adjust(wspace = 0.3)

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
    print("--- End Reading Data ---")

    try:
        w2vModel = Word2Vec.load(w2vModelPath)
    except FileNotFoundError:
        w2vModel = Word2Vec(trainComments + testComments, size = 200, min_count = 10, workers = 16)
        w2vModel.save(w2vModelPath)

    embeddingMatrix, word2Index = np.zeros((len(w2vModel.wv.vocab.items()) + 1, w2vModel.vector_size)), {}

    for i, (word, vec) in enumerate([(word, w2vModel.wv[word]) for word, _ in w2vModel.wv.vocab.items()]):
        embeddingMatrix[i + 1] = vec
        word2Index[word] = i + 1

    paddingLength = 50
    trainX = pad_sequences(np.array([[word2Index[word] if word in word2Index else 0 for word in sentence] for sentence in trainComments]), maxlen = paddingLength)
    trainY = trainLabels.reshape((-1, 1))
    print(trainX.shape, trainY.shape)

    model = Sequential()
    model.add(Embedding(input_dim = embeddingMatrix.shape[0],
                        output_dim = embeddingMatrix.shape[1],
                        weights = [embeddingMatrix],
                        trainable = False))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(128, activation = 'tanh')))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

    history = model.fit(trainX, trainY, batch_size = 480, epochs = 10, validation_split = 0.2)
    model.save(modelPath)

    testX = pad_sequences(np.array([[word2Index[word] if word in word2Index else 0 for word in sentence] for sentence in testComments]), maxlen = paddingLength)
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

