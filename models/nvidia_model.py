from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda, Activation, Dropout, Cropping2D
import tensorflow as tf
from image_preprocessing.size_manipulations import resize_image_32_32, resize_image_128_128
from image_preprocessing.color_manipulations import enhance_contrast
import os


def train_save(features, values, nb_epoch=5):
    print("Training the model using the NVIDIA architecture, performing {} epochs".format(nb_epoch))
    model = Sequential()

    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # model.add(Lambda(enhance_contrast))
    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae'])

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=128)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'nvidia_model.h5'))

    return model
