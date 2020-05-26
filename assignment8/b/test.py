import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import os
from keras.utils import multi_gpu_model
# os.environ["CUDA_VISIBLE_DEVICES"]="1";  
from PIL import Image
from skimage.transform import resize
import sys

#
from keras.utils import np_utils, generic_utils, to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras import backend as K
import keras
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.models import Model, load_model
from sklearn.utils import shuffle
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import optimizers
# K.set_image_dim_ordering('th')
#########
# Read from directory
#########

def read_data(folder):
    folders = os.listdir(folder)
    train_test_data = ImageDataGenerator(
            rescale=1./255,
            validation_split = 0.2)


    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_test_data.flow_from_directory(
            folder,
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical',
            subset = 'training',
            )

    validation_generator = train_test_data.flow_from_directory(
            folder,
            target_size=(100,100),
            batch_size=32,
            class_mode='categorical',
            subset = 'validation')
    return train_generator, validation_generator

  
  
  
train_generator, validation_generator = read_data(sys.argv[1])

model_final =load_model(sys.argv[2])



def train():
    score = model_final.evaluate_generator(train_generator)
    print('accuracy:', (score[1]))

train()
