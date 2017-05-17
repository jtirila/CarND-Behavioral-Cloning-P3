from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda, Activation, Dropout, Cropping2D
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from image_preprocessing.size_manipulations import resize_image_32_32, resize_image_128_128
from image_preprocessing.color_manipulations import enhance_contrast
import os


def train_save(features, values, nb_epoch=5):

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    features = features.astype('float32')
    datagen.fit(features)


    print("Training the model using the LeNet architecture, performing {} epochs".format(nb_epoch))

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 10), (0, 0)), input_shape=(160, 320, 3)))
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # model.add(Lambda(resize_image_128_128))
    # model.add(Conv2D(6, 5, 5, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), border_mode='same'))
    model.add(Lambda(enhance_contrast))
    model.add(Lambda(resize_image_32_32))
    model.add(Conv2D(6, 5, 5, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Conv2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(84, activation='tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='tanh'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    model.fit_generator(datagen.flow(features, values, batch_size=128, shuffle=True), samples_per_epoch=len(features) // 128, nb_epoch=nb_epoch)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'lenet_model.h5'))
    return model
