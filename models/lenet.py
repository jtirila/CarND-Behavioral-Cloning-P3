from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda, Activation, Dropout, Cropping2D
import tensorflow as tf
from image_preprocessing.size_manipulations import resize_image_32_32, resize_image_128_128
from image_preprocessing.color_manipulations import enhance_contrast
import os


def train_save(features, values, nb_epoch=5):
    print("Training the model using the LeNet architecture, performing {} epochs".format(nb_epoch))

    model = Sequential()
    model.add(Cropping2D(cropping=((60, 20), (20, 20)), input_shape=(160, 320, 3)))
    # Size now: (80, 280, 3)

    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Lambda(resize_image_32_32))
    model.add(Conv2D(6, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Conv2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=128)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'lenet_model.h5'))
    return model
