from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
import os


def train_save(train_features, train_values, nb_epoch=5):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(100, 320, 3)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.fit(train_features, train_values, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch)
    model.save(os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'output', 'simple_mlp.h5'))
    return model