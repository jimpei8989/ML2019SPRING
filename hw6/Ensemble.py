import sys, os, jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional

def Segment(filename):
    puctuations = set(list("，。？！～：（） "))
    df = pd.read_csv(filename)
    return [[s.lower() for s in sentence.split(' ') if s not in puctuations] for sentence in ' '.join(jieba.cut('\n'.join(df['comment']))).split('\n')]
    #  return [[s for s in jieba.cut(c) if s not in puctuations] for c in df['comment']]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dictTxt         = sys.argv[1]
    trainXPath      = sys.argv[2]
    trainYPath      = sys.argv[3]
    testXPath       = sys.argv[4]
    predictPath     = sys.argv[5]
    w2vModelPath    = sys.argv[6]

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

    #  models = ['GRU', 'LSTM', 'LSTM_2']
    models = ['GRU', 'GRU_2', 'LSTM', 'LSTM_2']
    #  models = ['GRU', 'GRU_2', 'GRU_D', 'LSTM', 'LSTM_2', 'LSTM_D']

    paddingLength = 50
    testX = pad_sequences(np.array([[word2Index[word] if word in word2Index else 0 for word in sentence] for sentence in testComments]), maxlen = paddingLength)

    outputY = np.concatenate([load_model('%s/model.h5' % model).predict(testX, batch_size = 480).reshape((-1, 1)) for model in models], axis = 1)
    signY = (outputY > 0.5)
    voteY = np.mean(signY, axis = 1)
    meanY = np.mean(outputY, axis = 1)
    print(outputY.shape, signY.shape)
    predictY = np.array([voteY[i] > 0.5 if voteY[i] != 0.5 else meanY[i] > 0.5 for i in range(testX.shape[0])])

    outputDF = pd.DataFrame(zip(range(testX.shape[0]), predictY.astype(int)), columns = ['id', 'label'])
    outputDF.to_csv(predictPath, index = None)

