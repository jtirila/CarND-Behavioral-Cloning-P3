from image_preprocessing.load_images import read_images_and_steering_angles
import numpy as np
from scipy.misc import imresize
import models.single_layer_dense as sld
import models.alexnet as alex
import models.lenet as lenet
import os
import sys

SIMPLE_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__),
        os.pardir, 
        os.pardir, 'data', 'batch_1')

MODEL_LOOKUP_DICT = {'alex': alex, 'sld': sld, 'lenet': lenet}


if __name__ == '__main__':
    print("Running {}".format(sys.argv[0]))
    print("Using data directory {}".format(sys.argv[1]))

    # Quitting if not using exactly one or two command line parameter
    assert len(sys.argv) in (2,3) 
    if len(sys.argv) == 3: 
        assert sys.argv[2] in ('lenet', 'alex', 'sld') 
        modelname = sys.argv[2]
    else: 
        modelname = 'alex'
        

    features, values = read_images_and_steering_angles(sys.argv[1])

    # Temporarily using a smaller set of training data
    features = features[:512]

    if modelname == 'alex':
        # Temporarily using scipy's imresize until I learn how this can be done in Keras
        features = np.array(list(map(lambda x: imresize(x, (227, 227)), features)))
    values = values[:512]

    # TODO: do something about the saved model name once all the models return it? 
    MODEL_LOOKUP_DICT[modelname].train_save(features, values)

