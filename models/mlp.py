from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda, Activation, Dropout
from keras import optimizers
import tensorflow as tf
import os


def train_save(features, values, nb_epoch=10):
    print("Training the model using a simple multilayer perceptron architecture, performing {} epochs".format(nb_epoch))
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten(input_shape=(100, 320, 3)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=128)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'mlp_model.h5'))
    return model
