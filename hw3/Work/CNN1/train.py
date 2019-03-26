import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def ReadTrainingData(path):
    df = pd.read_csv(path)
    Y = df['label'].values.reshape((-1, 1))
    print(df['features'][0])
