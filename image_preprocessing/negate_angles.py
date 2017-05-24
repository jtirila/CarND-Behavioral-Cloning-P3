import csv
import cv2
import os
import numpy as np


def negate_steering_angles(base_path):
    """Reads the driving log, multiplies each steering measurement by -1 and saves the log back
    
    :param base_path: A directory path where the csv file and the images are to be found
    :return: Nothing, just saves a new file with negeated angles."""

    lines = []
    with open(os.path.join(base_path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[3] = str(-1.0 * float(line[3]))
            lines.append(line)


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




