from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input


def train_save(features, values):
    inputs = Input(shape=(128, 128, 3))
    branch_1_conv_1 = Conv2D(48, 11, 11, border_mode='same')(inputs)
    branch_1_conv_2 = Conv2D(128, 5, 5, border_mode='same')(branch_1_conv_1)
    branch_1_maxpool_1 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(branch_1_conv_2)
    # TODO: connect this maxpool layer to both branches

    branch_2_conv_1 = Conv2D(48, 11, 11, border_mode='same')(inputs)
    branch_2_conv_2 = Conv2D(128, 5, 5, border_mode='same')(branch_2_conv_1)
    branch_2_maxpool_1 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(branch_2_conv_2)
    # TODO: connect this maxpool layer to both branches





