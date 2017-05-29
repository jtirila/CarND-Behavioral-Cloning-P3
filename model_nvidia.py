from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Lambda, Dropout, Cropping2D
from sklearn.utils import shuffle
import os
import sys
import csv
import numpy as np

import cv2


def run_nvidia_training(nb_epoch, data_path):

    print("About to load the images from files")
    features, values = read_images_and_steering_angles(data_path)
    print("steering angle max {}, min {}".format(max(values), min(values)))
    print("Finished loading the images from files")

    # Building the params for the train_save function, depending on whether number of epochs has been provided or not
    if nb_epoch is not None:
        params = [features, values, nb_epoch]
    else:
        params = [features, values]

    print("About to train and save the model.")
    features, values = shuffle(features, values)
    model = train_save(*params)

    # Shuffling again for the brief evaluation phase
    features, values = shuffle(features, values)
    print("About to evaluate the model")

    # Print some evaluation metrics from a small sample of the data
    model.evaluate(features[:128], values[:128])
    print("Finished evaluating the model")

    print("About to produce some sample predictions")
    for num in range(min(len(features), 48)):
        print("Prediction: {}, training value: {}".format(model.predict(np.array([features[num]])), format(values[num])))


def train_save(features, values, nb_epoch=5):
    """Trains the model, and saved it into a file.
    
    :param features: a numpy array containing the input images
    :param values: the steering angles, should be the same size as features and in same order
    :param nb_epoch: an optional epoch count to train.
    
    :return: The Keras model object."""

    print("Training the model using the NVIDIA architecture, performing {} epochs".format(nb_epoch))

    # Defining the model as in the lecture video, basically. Minor modifications to the cropping, dropout and
    # deep layer sizes
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 20), (5, 5)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.35))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    model.fit(features, values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=128)
    model.save('model.h5')

    return model


def read_images_and_steering_angles(base_path):
    """Reads the driving log and related images from the input directory
    
    :param base_path: A directory path where the csv file and the images are to be found
    :return: a tuple (images, measurements) where both elements are numpy arrays. The 
             images have been read using cv2, FIXME and at this point no conversion had been performed 
             from BRG to RGB."""

    lines = []
    with open(os.path.join(base_path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    img_base_path = os.path.join(base_path, 'IMG')
    for line in lines:
        source_path = line[0]
        filename = source_path.split("/")[-1]
        current_path = os.path.join(img_base_path, filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(float(line[3]))
    return np.array(images), np.array(measurements)


# NOTE: This method is not used in the model as such but was rather used in data augmentation phase. Included for
# NOTE: completeness
def negate_steering_angles(base_path):
    """Reads the driving log, multiplies each steering measurement by -1 and saves the log back. 
    
    NOTE: this was a one-off script using hard coded filenames and cannot be reused as such. 
    
    :param base_path: A directory path where the csv file and the images are to be found
    :return: Nothing, just saves a new file with negeated angles."""

    lines = []

    # Inverting the angles
    with open(os.path.join(base_path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[3] = str(-1.0 * float(line[3]))
            lines.append(line)

    # Renaming the files in driving_log
    with open(os.path.join(base_path, 'driving_log_2.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for line in lines:
            source_path = line[0]
            filename = source_path.split("/")[-1]
            new_filename = filename.split(".")[-2] + "_flipped." + filename.split(".")[-1]
            new_path = "/".join(source_path.split("/")[:-1] + [new_filename])
            os.rename(os.path.join(base_path, "IMG", filename), os.path.join(base_path, "IMG", new_filename))
            line[0] = new_path
            writer.writerow(line)


if __name__ == "__main__":
    # Some debug prints
    print("Running {}".format(sys.argv[0]))
    print("All args: {}".format(sys.argv))

    # Quitting if not using exactly one or two command line parameter
    assert len(sys.argv) in (2, 3)

    print("Using data directory {}".format(sys.argv[1]))
    nb_epoch = None
    if len(sys.argv) > 2:
        nb_epoch = int(sys.argv[2])

    run_nvidia_training(nb_epoch, sys.argv[1])
