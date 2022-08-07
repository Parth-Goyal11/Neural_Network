import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

dataset = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = dataset.load_data()

'''plt.imshow(x_train[0], cmap=plt.cm.binary)     Uncomment to show image, 
                                                    Delete second parameter to view image in color
plt.show()
print(x_train[0])                      Uncomment to show 28X28 grayscale array of pixels'''

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_train, axis=1)

SQUARE_IMG_SIZE = 28
x_train_new = np.array(x_train).reshape(-1, SQUARE_IMG_SIZE, SQUARE_IMG_SIZE, 1)
x_test_new = np.array(x_test).reshape(-1, SQUARE_IMG_SIZE, SQUARE_IMG_SIZE, 1)

model_shape = x_train_new.shape
print(model_shape)

mod = Sequential()

# Layer 1(Input Layer)
mod.add(Conv2D(64, (3, 3), input_shape=x_train_new.shape[1:]))  # Instantiate 64 3X3 kernels
mod.add(Activation("relu"))  # Use the relu function for activation, or condensing value between 0-1
mod.add(MaxPooling2D(pool_size=(2, 2)))  # Creates a 2 by 2 matrix and only passes on the maximum value in that matrix

# Layer 2
mod.add(Conv2D(64, (3, 3)))
mod.add(Activation("relu"))
mod.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
mod.add(Conv2D(64, (3, 3)))
mod.add(Activation("relu"))
mod.add(MaxPooling2D(pool_size=(2, 2)))

# Make the Model 1D
mod.add(Flatten())
# Start Layer of Fully Connected Neurons
mod.add(Dense(64))
mod.add(Activation("relu"))

# Layer 2 of Connected Neurons
mod.add(Dense(32))
mod.add(Activation("relu"))

# Last Layer of Neurons
mod.add(Dense(10))  # 10 Different Possible Digits
mod.add(Activation('softmax'))

#Train Function
mod.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
mod.fit(x_train_new, y_train, epochs=5, validation_split=0.25)  # 25 percent of the training data is used for validation

#Iterate through test data and determine how accurate the neural network is
test_loss, test_acc = mod.evaluate(x_test_new, y_test)
print("Loss on Test: " + str(test_loss))
print("Model Accuracy: " + str(test_acc))


predictions = mod.predict([x_test_new])
#Uncomment this to test on a specific image in the dataset

'''
plt.imshow(x_test[0])                     Use this to bring up an actual image in the dataset
plt.show()
print(np.argmax(predictions[0]))          This will print out the prediction that the neural network's prediction'''


   
