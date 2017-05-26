from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda, Activation, Dropout, Cropping2D
import tensorflow as tf
from image_preprocessing.size_manipulations import resize_image_32_32, resize_image_128_128
from image_preprocessing.color_manipulations import enhance_contrast
import os


def train_save(features, values, nb_epoch=5):
    print("Training the model using my very own architecture, inspired by lenet, performing {} epochs".format(nb_epoch))

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 10), (10, 10)), input_shape=(160, 320, 3)))
    # Size now: (100, 300, 3)

    # model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # model.add(Lambda(resize_image_128_128))

    model.add(Conv2D(6, 37, 45, border_mode='valid'))
    # size now: (64, 256, 6)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 4), border_mode='valid'))
    # size now: (32, 64, 6)

    model.add(Conv2D(6, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), border_mode='same'))
    # size now: (32, 32, 6)

    model.add(Conv2D(6, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Conv2D(16, 5, 5, border_mode='valid'))
    # Size: 28x28
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=128)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'lenet_mod_model.h5'))
    return model
