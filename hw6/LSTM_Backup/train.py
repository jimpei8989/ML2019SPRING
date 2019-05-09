import sys, os, jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from keras.callbacks import ReduceLROnPlateau

def Segment(filename):
    puctuations = set(list("，。？！～：（） "))
    df = pd.read_csv(filename)
    return [[s.lower() for s in sentence.split(' ') if s not in puctuations] for sentence in ' '.join(jieba.cut('\n'.join(df['comment']))).split('\n')]
    #  return [[s for s in jieba.cut(c) if s not in puctuations] for c in df['comment']]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    dictTxt         = sys.argv[1]
    trainXPath      = sys.argv[2]
    trainYPath      = sys.argv[3]
    testXPath       = sys.argv[4]
    predictPath     = sys.argv[5]
    w2vModelPath    = sys.argv[6]
    modelPath       = sys.argv[7]

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
                        trainable = True))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(128, activation = 'tanh')))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    model.fit(trainX, trainY, batch_size = 480, epochs = 5, validation_split = 0.2, callbacks = [ReduceLR])
    model.save(modelPath)

    testX = pad_sequences(np.array([[word2Index[word] if word in word2Index else 0 for word in sentence] for sentence in testComments]), maxlen = paddingLength)
    predictY = model.predict(testX, batch_size = 480)

    outputDF = pd.DataFrame(zip(range(testX.shape[0]), (predictY.reshape(-1) > 0.5).astype(int)), columns = ['id', 'label'])
    outputDF.to_csv(predictPath, index = None)

