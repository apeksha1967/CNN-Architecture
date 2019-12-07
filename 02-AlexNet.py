import tflearn
from keras.utils import np_utils
import tflearn.datasets.oxflower17 as oxflower17
x, y = oxflower17.load_data(one_hot=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalization
#x_train /= 255
#x_test /= 255

from keras.models import Sequential
from keras import models, layers
import keras
import numpy as np

np.random.seed(1000)

#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(layers.Conv2D(filters=96, activation='relu', input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
# Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(layers.Conv2D(filters=256, activation='relu', kernel_size=(11,11), strides=(1,1), padding='valid'))
# Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(layers.Conv2D(filters=384, activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid'))

# 4th Convolutional Layer
model.add(layers.Conv2D(filters=384, activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid'))

# 5th Convolutional Layer
model.add(layers.Conv2D(filters=256, activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid'))
# Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(layers.Flatten())
# 1st Fully Connected Layer
model.add(layers.Dense(4096, activation='relu', input_shape=(224*224*3,)))
# Add Dropout to prevent overfitting
model.add(layers.Dropout(0.4))

# 2nd Fully Connected Layer
model.add(layers.Dense(4096, activation='relu'))
# Add Dropout
model.add(layers.Dropout(0.4))

# 3rd Fully Connected Layer
model.add(layers.Dense(1000, activation='relu'))
# Add Dropout
model.add(layers.Dropout(0.4))

# Output Layer
model.add(layers.Dense(17, activation='softmax'))

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