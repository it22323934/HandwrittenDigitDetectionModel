import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# Normalise the data
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

#Convulutional Neural Network
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# Relu is the activation function stands for Rectified Linear Unit
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
#makes sure all the 10 neurons add up to 1 this is why we use softmax. Softmax gives the probablity of each output(digit)
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model. Epochs is the number of times the model will see the data
model.fit(x_train,y_train,epochs=20)
model.save('handwrittenRecogModel2.keras')

