from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, merge, Reshape
import os

CONCAT_AXIS = 1


def train_save(features, values):
    inputs = Input(shape=(224, 224, 3))
    resized_inputs = Reshape(input_shape=(160, 320), target_shape=(224, 224))(inputs)
    branch_1_conv_1 = Conv2D(48, 11, 11, border_mode='same')(resized_inputs)
    branch_1_maxpool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(branch_1_conv_1)
    branch_1_conv_2 = Conv2D(128, 5, 5, border_mode='same')(branch_1_maxpool_1)
    branch_1_maxpool_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(branch_1_conv_2)

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

    branch_1_dense_1 = Dense(2048)(merge_2)
    branch_2_dense_1 = Dense(2048)(merge_2)

    merge_3 = merge(inputs=[branch_1_dense_1, branch_2_dense_1], mode='concat', concat_axis=CONCAT_AXIS)

    branch_2_dense_2 = Dense(1)(merge_3)
    branch_1_dense_2 = Dense(1)(merge_3)

    merge_4 = merge(inputs=[branch_1_dense_2, branch_2_dense_2], mode='concat', concat_axis=CONCAT_AXIS)

    output = Dense(1)(merge_4)

    model = Model(input=inputs, output=output)
    model.compile(loss='mse', optimizer='adam')

    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'alexnet_model.h5'))






