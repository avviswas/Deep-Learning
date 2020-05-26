from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
import numpy as np
import keras	
import sys
from keras import regularizers
from keras.regularizers import l2
from keras.utils import plot_model
from keras import backend as K
K.set_image_dim_ordering('th')



batch_size = 128
nb_classes = 10
nb_epoch = 100

img_channels = 3
img_rows = 112
img_cols = 112

X_train = np.load(sys.argv[1])
# X_test = np.load('image_net/x_test.npy')
Y_train = np.load(sys.argv[2])
# Y_test = np.load('image_net/y_test.npy')

print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

print('Y_train shape:', Y_train.shape)
# print('Y_test shape:', Y_test.shape)

#Y_train = keras.utils.to_categorical(Y_train, nb_classes)
#Y_test = keras.utils.to_categorical(Y_test, nb_classes)#convert label into one-hot vector
Y_train = to_categorical(Y_train, nb_classes)
# Y_test = to_categorical(Y_test, nb_classes)#convert label into one-hot vector

print('Y_train shape:', Y_train.shape)
# print('Y_test shape:', Y_test.shape)

#exit()

model = Sequential()


# model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

#Layer 1
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation="relu", input_shape=(3,112,112), kernel_regularizer=regularizers.l2(0.0005)))#Convo$
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))
#Layer 2
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))#Convo$
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))
#Layer 3
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))#Convo$
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))
#Layer 4
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))
#layer 5
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))

#layer 6


#Dense layer
model.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(Dense(10))#Fully connected layer
model.add(Activation('softmax'))

keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=False, cpu_relocation=False)

#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
# opt = keras.optimizers.SGD(lr=0.0005, decay=1e-6)
# opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)# best one
# opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.save('my_model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def train():
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True)
    model.save(sys.argv[3])

train()
# plot_model(model, to_file='model.png')

