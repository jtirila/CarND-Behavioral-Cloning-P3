from image_preprocessing.load_images import read_images_and_steering_angles
import numpy as np
from scipy.misc import imresize
import models.single_layer_dense as sld
import models.alexnet as alex
import os

SIMPLE_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'batch_1')


if __name__ == '__main__':
    features, values = read_images_and_steering_angles(SIMPLE_DATA_BASE_PATH)
    features = features[:512]
    features = np.array(list(map(lambda x: imresize(x, (227, 227)), features)))
    values = values[:512]
    model = alex
    # model = sld
    # model = lenet
    model.train_save(features, values)
