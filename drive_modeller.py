import os
import sys

from matplotlib import pyplot as plt

import numpy as np
from scipy.misc import imresize
from sklearn.utils import shuffle

import models.alexnet as alex
import models.lenet as lenet
import models.mlp as mlp
import models.single_layer_dense as simple
import models.simple_mlp as simple_mlp
import models.single_layer_dense as sld
from image_preprocessing.load_images import read_images_and_steering_angles

SIMPLE_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__),
        os.pardir, 
        os.pardir, 'data', 'batch_1')

MODEL_LOOKUP_DICT = {'alex': alex, 'simple': simple, 'simple_mlp': simple_mlp, 'sld': sld, 'lenet': lenet, 'mlp': mlp}


if __name__ == '__main__':
    print("Running {}".format(sys.argv[0]))
    print("All args: {}".format(sys.argv))
    assert len(sys.argv) in (2, 3, 4)
    print("Using data directory {}".format(sys.argv[1]))

    # Quitting if not using exactly one or two command line parameter
    nb_epoch = None
    if len(sys.argv) > 2:
        assert sys.argv[2] in MODEL_LOOKUP_DICT.keys()
        modelname = sys.argv[2]
        if len(sys.argv) == 4:
            nb_epoch = int(sys.argv[3])
    else:
        modelname = 'alex'

    print("About to load the images from files")
    features, values = read_images_and_steering_angles(sys.argv[1])
    print("steering angle max {}, min {}".format(max(values), min(values)))

    print("Finished loading the images from files")

    # Temporarily using a smaller set of training data
    # features = features[:256]
    # values = values[:256]

    # features = np.array(list(map(lambda x: x[60:, :, :], features)))

    if nb_epoch is not None:
        params = [features, values, nb_epoch]
    else:
        params = [features, values]
    # TODO: do something about the saved model name once all the models return it?
    print("About to train and save the model.")
    features, values = shuffle(features, values)
    model = MODEL_LOOKUP_DICT[modelname].train_save(*params)
    features, values = shuffle(features, values)
    print("About to evaluate the model")

    # TODO: this is basically just PoC / mockup code.
    model.evaluate(features[:128], values[:128])
    print("Finished evaluating the model")

    print("About to produce some sample predictions")
    for num in range(min(len(features), 128)):
        print("Prediction: {}, training value: {}".format(model.predict(np.array([features[num]])), format(values[num])))
