from keras.models import Sequential
from keras.layers import Dense, Flatten
import os


def train_save(train_features, train_values, nb_epoch=5):
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.fit(train_features, train_values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'simple_model.h5'))
    return model