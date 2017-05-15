from image_preprocessing.load_images import read_images_and_steering_angles
import numpy as np
from scipy.misc import imresize
import models.single_layer_dense as sld
import models.mlp as mlp
import models.alexnet as alex
import models.lenet as lenet
from sklearn.utils import shuffle
import os
import sys

SIMPLE_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__),
        os.pardir, 
        os.pardir, 'data', 'batch_1')

MODEL_LOOKUP_DICT = {'alex': alex, 'sld': sld, 'lenet': lenet, 'mlp': mlp}


if __name__ == '__main__':
    print("Running {}".format(sys.argv[0]))
    print("All args: {}".format(sys.argv))
    assert len(sys.argv) in (2,3)
    print("Using data directory {}".format(sys.argv[1]))

    # Quitting if not using exactly one or two command line parameter
    if len(sys.argv) == 3:
        assert sys.argv[2] in MODEL_LOOKUP_DICT.keys()
        modelname = sys.argv[2]
    else: 
        modelname = 'alex'

    print("About to load the images from files")
    features, values = read_images_and_steering_angles(sys.argv[1])
    print("Finished loading the images from files")

    # Temporarily using a smaller set of training data
    # features = features[:256]
    # values = values[:256]

    if modelname == 'alex':
        print("About to resize images")
        # Temporarily using scipy's imresize until I learn how this can be done in Keras
        features = np.array(list(map(lambda x: imresize(x, (227, 227)), features)))
        print("Finished resizing images")
    elif modelname == 'lenet':
        print("About to resize images")
        # Temporarily using scipy's imresize until I learn how this can be done in Keras
        features = np.array(list(map(lambda x: imresize(x, (32, 32)), features)))
        print("Finished resizing images")

    # TODO: do something about the saved model name once all the models return it?
    print("About to train and save the model.")
    model = MODEL_LOOKUP_DICT[modelname].train_save(features, values)
    features, values = shuffle(features, values)
    print("About to evaluate the model")
    model.evaluate(features[:128], values[:128])
    print("Finished evaluating the model")

    print("About to produce some sample predictions")
    print(model.predict(features[:64]))

