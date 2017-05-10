from image_preprocessing.load_images import read_images_and_steering_angles
import models.single_layer_dense as sld
import models.alexnet as alex
import os

SIMPLE_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'batch_1')


if __name__ == '__main__':
    features, values = read_images_and_steering_angles(SIMPLE_DATA_BASE_PATH)
    # sld.train_save(features, values)
    alex.train_save(features, values)

