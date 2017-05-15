from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Lambda, Merge, Reshape, RepeatVector
from keras import optimizers
import tensorflow as tf
import os

CONCAT_AXIS = 3


def train_save(features, values, nb_epochs=5):
    print("Training the model using the AlexNet architecture")

    input_layer = Sequential()
    input_layer.add(Lambda(lambda x: x, input_shape=(227, 227, 3)))

    branch_1_1 = Sequential()
    branch_2_1 = Sequential()

    branch_1_1.add(input_layer)
    branch_2_1.add(input_layer)

    branch_1_1.add(Conv2D(48, 11, 11, border_mode='same'))
    branch_1_1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    branch_1_1.add(Conv2D(128, 5, 5, border_mode='same'))
    branch_1_1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    branch_2_1.add(Conv2D(48, 11, 11, border_mode='same'))
    branch_2_1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    branch_2_1.add(Conv2D(128, 5, 5, border_mode='same'))
    branch_2_1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    merge_1 = Merge([branch_1_1, branch_2_1], mode='concat')

    branch_1_2 = Sequential()
    branch_1_2.add(merge_1)
    branch_2_2 = Sequential()
    branch_2_2.add(merge_1)

    branch_1_2.add(Conv2D(192, 3, 3, border_mode='same'))
    branch_1_2.add(Conv2D(192, 3, 3, border_mode='same'))
    branch_1_2.add(Conv2D(128, 3, 3, border_mode='same'))
    branch_1_2.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), border_mode='same'))

    branch_2_2.add(Conv2D(192, 3, 3, border_mode='same'))
    branch_2_2.add(Conv2D(192, 3, 3, border_mode='same'))
    branch_2_2.add(Conv2D(128, 3, 3, border_mode='same'))
    branch_2_2.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), border_mode='same'))

    merge_2 = Merge([branch_1_2, branch_2_2], mode='concat')

    flat_layer = Sequential()
    flat_layer.add(merge_2)
    flat_layer.add(Flatten())

    branch_1_3 = Sequential()
    branch_1_3.add(flat_layer)
    branch_2_3 = Sequential()
    branch_2_3.add(flat_layer)

    branch_1_3.add(Dense(2048, activation='relu'))
    branch_1_3.add(Dense(2048, activation='relu'))

    branch_2_3.add(Dense(2048, activation='relu'))
    branch_2_3.add(Dense(2048, activation='relu'))

    merge_3 = Merge([branch_1_3, branch_2_3], mode='concat')

    branch_1_4 = Sequential()
    branch_1_4.add(merge_3)
    branch_2_4 = Sequential()
    branch_2_4.add(merge_3)

    branch_1_4.add(Dense(2048, activation='relu'))
    branch_2_4.add(Dense(2048, activation='relu'))

    merge_4 = Merge([branch_1_4, branch_2_4], mode='concat')

    output = Sequential()
    output.add(merge_4)
    output.add(Dense(1, activation='relu'))

    output.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    output.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epochs, batch_size=24)
    output.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'alexnet_model.h5'))

    return output


