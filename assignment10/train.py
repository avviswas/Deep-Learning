from keras.datasets import cifar10
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
import numpy as np
import keras
import tensorflow as tf
from keras import backend as k
from numpy.random import randn, randint
import sys

(train_X, train_Y), (x_test, y_test) = cifar10.load_data()

idx = (np.logical_or(train_Y == 0, train_Y == 1)).reshape(train_X.shape[0])
train_X = train_X[idx]
train_Y = train_Y[np.logical_or(train_Y == 0, train_Y == 1)]
train_Y = keras.utils.to_categorical(train_Y, 2)

mean_X = np.mean(train_X[:10,:], axis=0)
train_X = train_X - mean_X

model = load_model('mlp20node_model.h5')
model2 = Sequential()
model2.add(Flatten(input_shape = train_X.shape[1:]))
#inputs = keras.layers.Input(shape=train_X.shape[1:])
model2.add(Dense(100))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(100))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(2, activation='softmax'))
#model2 = Model(inputs=inputs, outputs=outputs)
opt = SGD(lr=0.0001)
model2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

epocs=5
train_X1 = train_X[:10,:]
for epoch in range(5):
      print(train_X1.shape)
      train_Y1 = model.predict_classes(train_X1)
      train_Y1 = keras.utils.to_categorical(train_Y1.reshape(-1, 1), 2)
      epoc_steps = int(train_X1.shape[0]/epocs)
      model2.fit(train_X1,train_Y1, steps_per_epoch=epoc_steps, epochs=epocs)
      grads = k.gradients(model2.output, model2.input)[0]
      s = tf.compat.v1.Session()
      iterate = k.function(model2.input, [grads])
      grad = iterate(train_X1)
      #if randint(0, 2) == 1:
      grad= train_X1 + 0.5*np.sign(grad)
      #else:
        #grad= train_X1 - 0.1*np.sign(grad)
      grad = grad.reshape(train_X1.shape[0],32,32,3)
      
      train_X1 = np.append(train_X1, grad, axis=0)
model2.save(sys.argv[1])
