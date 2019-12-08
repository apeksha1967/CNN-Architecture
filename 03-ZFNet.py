import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
import pickle, datetime
import preprocess as pp

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import  PIL.Image
import tflearn
import tflearn.datasets.oxflower17 as oxflower17

x, y = oxflower17.load_data(one_hot=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalization
#x_train /= 255
#x_test /= 255

DROPOUT = 0.5
model_input = Input(shape = (224, 224, 3))

# First convolutional Layer (96x7x7)
z = Conv2D(filters = 96, kernel_size = (7,7), strides = (2,2), activation = "relu")(model_input)
z = ZeroPadding2D(padding = (1,1))(z)
z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)
z = BatchNormalization()(z)

# Second convolutional Layer (256x5x5)
z = Convolution2D(filters = 256, kernel_size = (5,5), 
                  strides = (2,2), activation = "relu")(z)
z = ZeroPadding2D(padding = (1,1))(z)
z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)
z = BatchNormalization()(z)

# Rest 3 convolutional layers
z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 384, kernel_size = (3,3), 
                  strides = (1,1), activation = "relu")(z)

z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 384, kernel_size = (3,3), 
                  strides = (1,1), activation = "relu")(z)

z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 256, kernel_size = (3,3), 
                  strides = (1,1), activation = "relu")(z)

z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)
z = Flatten()(z)

z = Dense(4096, activation="relu")(z)
z = Dropout(DROPOUT)(z)

z = Dense(4096, activation="relu")(z)
z = Dropout(DROPOUT)(z)

z = Dense(1000, activation="relu")(z)
z = Dropout(DROPOUT)(z)

model_output = Dense(17, activation="softmax")(z)

model = Model(model_input, [model_output])
model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

m = model.fit(x=x_train, y=y_train, epochs=10, batch_size=64, validation_data=(x_test,y_test),verbose=1)

test_score = model.evaluate(x_test, y_test)
print("Test loss: {}, accuracy: {}".format(test_score[0], test_score[1]*100))

import matplotlib.pyplot as plt

# Accuracy graph
fig, ax = plt.subplots()
ax.plot(m.history['acc'], 'o-')
ax.plot(m.history['val_acc'], 'x-')

ax.legend(['Train acc', 'Validation acc'], loc = 0)
ax.set_title("Train/Validation accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")

# Loss graph
fig, ax = plt.subplots()
ax.plot(m.history['loss'], 'o-')
ax.plot(m.history['val_loss'], 'x-')
ax.legend(['Train loss', 'Validation loss'], loc = 0)
ax.set_title("Train/Validation loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")