from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalization
x_train /= 255
x_test /= 255

#One hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#convert into 4D array
#because we had data in form 60000x28x28(training)
#as per LeNet CNN it should be converted to 60000x28x28x1
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

from keras.models import Sequential
from keras import models, layers
import keras

model = Sequential()

#C1 - Convolution layer
model.add(layers.Conv2D(6,kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=(28,28,1), padding='same'))

#Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

#C2 - Convolutional layer
model.add(layers.Conv2D(16,kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid'))

# Pooling layer
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Fully connected layer
model.add(layers.Conv2D(120,kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid'))

# Flatten the CNN
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(84, activation='tanh'))

#Output layer with softmax
model.add(layers.Dense(10, activation='softmax'))


# compile model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])

m = model.fit(x=x_train, y=y_train, epochs=10, batch_size=128, validation_data=(x_test,y_test),verbose=1)

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