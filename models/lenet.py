from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda, Activation
from keras import optimizers
import tensorflow as tf
import os


def train_save(features, values, nb_epoch=5):
    print("Training the model using the LeNet architecture")
    model = Sequential()
    model.add(Conv2D(6, 5, 5, input_shape=(32, 32, 3), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='same'))
    model.add(Conv2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    model.add(Activation('tanh'))

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=128)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'lenet_model.h5'))
    return model
