from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, merge, Lambda
from keras import optimizers
import tensorflow as tf
import os

CONCAT_AXIS = 3


def train_save(features, values, nb_epochs=5):
    print("Training the model using the AlexNet architecture")
    inputs = Input(shape=(227, 227, 3))
    # resized_inputs = Lambda(lambda x: tf.image.resize_images(x, (227, 227)))(inputs)
    # branch_1_conv_1 = Conv2D(48, 11, 11, border_mode='same')(resized_inputs)
    branch_1_conv_1 = Conv2D(48, 11, 11, border_mode='same')(inputs)
    branch_1_maxpool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(branch_1_conv_1)
    branch_1_conv_2 = Conv2D(128, 5, 5, border_mode='same')(branch_1_maxpool_1)
    branch_1_maxpool_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(branch_1_conv_2)

    # branch_2_conv_1 = Conv2D(48, 11, 11, border_mode='same')(resized_inputs)
    branch_2_conv_1 = Conv2D(48, 11, 11, border_mode='same')(inputs)
    branch_2_maxpool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(branch_2_conv_1)
    branch_2_conv_2 = Conv2D(128, 5, 5, border_mode='same')(branch_2_maxpool_1)
    branch_2_maxpool_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(branch_2_conv_2)

    merge_1 = merge(inputs=[branch_1_maxpool_2, branch_2_maxpool_2], mode='concat', concat_axis=CONCAT_AXIS)

    branch_1_conv_3 = Conv2D(192, 3, 3, border_mode='same')(merge_1)
    branch_1_conv_4 = Conv2D(192, 3, 3, border_mode='same')(branch_1_conv_3)
    branch_1_conv_5 = Conv2D(128, 3, 3, border_mode='same')(branch_1_conv_4)
    branch_1_maxpool_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), border_mode='same')(branch_1_conv_5)

    branch_2_conv_3 = Conv2D(192, 3, 3, border_mode='same')(merge_1)
    branch_2_conv_4 = Conv2D(192, 3, 3, border_mode='same')(branch_2_conv_3)
    branch_2_conv_5 = Conv2D(128, 3, 3, border_mode='same')(branch_2_conv_4)
    branch_2_maxpool_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), border_mode='same')(branch_2_conv_5)

    merge_2 = merge(inputs=[branch_1_maxpool_3, branch_2_maxpool_3], mode='concat', concat_axis=CONCAT_AXIS)
    flat1 = Flatten()(merge_2)

    branch_1_dense_1 = Dense(2048, activation='relu')(flat1)
    branch_2_dense_1 = Dense(2048, activation='relu')(flat1)

    merge_3 = merge(inputs=[branch_1_dense_1, branch_2_dense_1], mode='concat', concat_axis=0)

    branch_2_dense_2 = Dense(2048, activation='relu')(merge_3)
    branch_1_dense_2 = Dense(2048, activation='relu')(merge_3)

    merge_4 = merge(inputs=[branch_1_dense_2, branch_2_dense_2], mode='concat', concat_axis=0)

    output = Dense(1, activation='relu')(merge_4)

    model = Model(input=inputs, output=output)
    # sgd = optimizers.SDG(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epochs, batch_size=128)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'alexnet_model.h5'))

    return model


