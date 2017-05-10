import csv
import cv2
import os
import numpy as np


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




