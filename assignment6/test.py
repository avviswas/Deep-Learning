import numpy as np
import keras
import sys
import h5py
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import backend as K


K.set_image_dim_ordering('th')


classes = 10
X = sys.argv[1]
X_test = np.load(X)
Y = sys.argv[2]
Y_test = np.load(Y)
Y_test = to_categorical(Y_test, classes)
model_file = sys.argv[3]
model = load_model(model_file)

def score():
    score = model.evaluate(X_test,Y_test)
    print(model.metrics_names[1], score[1]*100)

score()

