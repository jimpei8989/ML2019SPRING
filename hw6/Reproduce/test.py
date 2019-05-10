import sys, os, jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

def Segment(filename):
    puctuations = set(list("，。？！～：（） "))
    df = pd.read_csv(filename)
    return [[s.lower() for s in sentence.split(' ') if s not in puctuations] for sentence in ' '.join(jieba.cut('\n'.join(df['comment']))).split('\n')]
    #  return [[s for s in jieba.cut(c) if s not in puctuations] for c in df['comment']]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    dictTxt         = sys.argv[1]
    testXPath       = sys.argv[2]
    predictPath     = sys.argv[3]
    w2vModelPath    = sys.argv[4]
    modelPath       = sys.argv[5]

    jieba.load_userdict(dictTxt)
    jieba.enable_parallel()

    print("--- Begin Reading Data ---")
    testComments = Segment(testXPath)
    print("--- End Reading Data ---")

    w2vModel = Word2Vec.load(w2vModelPath)
    model = load_model(modelPath)

    paddingLength = 50
    embeddingMatrix, word2Index = np.zeros((len(w2vModel.wv.vocab.items()) + 1, w2vModel.vector_size)), {}
    for i, (word, vec) in enumerate([(word, w2vModel.wv[word]) for word, _ in w2vModel.wv.vocab.items()]):
        embeddingMatrix[i + 1] = vec
        word2Index[word] = i + 1

    testX = pad_sequences(np.array([[word2Index[word] if word in word2Index else 0 for word in sentence] for sentence in testComments]), maxlen = paddingLength)
    predictY = model.predict(testX, batch_size = 480)

    outputDF = pd.DataFrame(zip(range(testX.shape[0]), (predictY.reshape(-1) > 0.5).astype(int)), columns = ['id', 'label'])
    outputDF.to_csv(predictPath, index = None)

