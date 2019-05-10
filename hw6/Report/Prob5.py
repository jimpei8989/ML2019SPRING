import sys, os, jieba, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from keras.callbacks import ReduceLROnPlateau

def Segment(sentences):
    return [[seg for seg in jieba.cut(sentence)] for sentence in sentences]

def S2V(s, wordMap):
    ret = np.zeros(len(wordMap) + 1)
    for w in s:
        try:
            ret[wordMap[w]] += 1
        except KeyError:
            ret[0] += 1
    return ret.reshape((1, -1))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    dictTxt         = "../data/dict.txt.big"
    w2vModelPath    = "Prob1/word2vec.model"
    wordMapPath     = "Prob2/wordMap.pkl"
    RNNPath         = "Prob1/model.h5"
    BOWPath         = "Prob2/model.h5"

    jieba.load_userdict(dictTxt)
    jieba.enable_parallel()

    sentences = ['在說別人白痴之前，先想想自己', '在說別人之前先想想自己，白痴']
    segmented = Segment(sentences)

    # RNN
    w2vModel = Word2Vec.load(w2vModelPath)
    RNN = load_model(RNNPath)
    embeddingMatrix, word2Index = np.zeros((len(w2vModel.wv.vocab.items()) + 1, w2vModel.vector_size)), {}
    for i, (word, vec) in enumerate([(word, w2vModel.wv[word]) for word, _ in w2vModel.wv.vocab.items()]):
        embeddingMatrix[i + 1] = vec
        word2Index[word] = i + 1

    paddingLength = 50
    testX = pad_sequences(np.array([[word2Index[word] if word in word2Index else 0 for word in sentence] for sentence in segmented]), maxlen = paddingLength)
    predictY = RNN.predict(testX)
    print('RNN Output')
    print(predictY)

    # BOW
    with open(wordMapPath, 'rb') as f:
        wordMap = pickle.load(f)
    BOW = load_model(BOWPath)
    testX = np.concatenate([S2V(sentence, wordMap) for sentence in segmented], axis = 0)
    predictY = BOW.predict(testX)
    print('BOW Output')
    print(predictY)

