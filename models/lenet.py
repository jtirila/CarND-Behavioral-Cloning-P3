from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda, Activation
import tensorflow as tf
import os


def train_save(features, values, nb_epoch=5):
    print("Training the model using the LeNet architecture, performing {} epochs".format(nb_epoch))

    model = Sequential()
    model.add(Lambda(lambda x: tf.image.resize_images(x, (32, 32)), input_shape=(100, 320, 3)))
    model.add(Conv2D(6, 5, 5, input_shape=(32, 32, 3), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Conv2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=128)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'lenet_model.h5'))
    return model
