from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda, Activation, Dropout
from keras import optimizers
import tensorflow as tf
import os


def train_save(features, values, nb_epoch=10):
    print("Training the model using a simple multilayer perceptron architecture")
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(500, activation='tanh'))
    model.add(Dense(400, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(80, activation='tanh'))
    model.add(Dense(60, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='tanh'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(6, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='tanh'))
    model.add(Dense(1))

    model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mae'])

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=128)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'mlp_model.h5'))
    return model
