from image_preprocessing.load_images import read_images_and_steering_angles
import numpy as np
from scipy.misc import imresize
import models.single_layer_dense as sld
import models.alexnet as alex
import os
import sys

SIMPLE_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__),
        os.pardir, 
        os.pardir, 'data', 'batch_1')


if __name__ == '__main__':
    print("Argv 0: {}".format(sys.argv[0]))
    assert len(sys.argv) == 2
    features, values = read_images_and_steering_angles(sys.argv[1])
    features = features[:512]
    features = np.array(list(map(lambda x: imresize(x, (227, 227)), features)))
    values = values[:512]
    model = alex
    # model = sld
    # model = lenet
    # model.train_save(features, values)
