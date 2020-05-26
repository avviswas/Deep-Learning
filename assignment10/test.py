from keras.datasets import cifar10
from keras.models import load_model
from numpy.random import randn, randint
import keras
import tensorflow as tf
from keras import backend as k
import numpy as np
import sys

(train_X, train_Y), (x_test, y_test) = cifar10.load_data()
idx = (np.logical_or(train_Y == 0, train_Y == 1)).reshape(train_X.shape[0])
train_X = train_X[idx]
train_Y = train_Y[np.logical_or(train_Y == 0, train_Y == 1)]
train_Y = keras.utils.to_categorical(train_Y, 2)

model2 = load_model(sys.argv[1])
train_X2 = train_X[11:,:]
train_Y2 = train_Y[11:,:]
test_m = np.mean(train_X2, axis=0)
xtrain2 = train_X2 - test_m
grads = k.gradients(model2.output, model2.input)[0]
s1 = tf.compat.v1.Session()
iterate1 = k.function(model2.input, [grads])
grad = iterate1(train_X2)
grad = train_X2 + 0.0625 * np.sign(grad)
grad = grad.reshape(train_X2.shape[0],32,32,3)
scores = model2.evaluate(grad,train_Y2)
print("accuracy: ",scores[1])
