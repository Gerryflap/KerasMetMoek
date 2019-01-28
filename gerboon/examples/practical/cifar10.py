import keras as ks
import numpy as np
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def to_one_hot(x):
    c = np.zeros(x.shape[0], 10)
    c[np.arange(x.shape[0]), x] = 1
    return c


y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

model = ks.models.Sequential()
model.add(ks.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)))
model.add(ks.layers.Conv2D(64, 3, activation='relu'))
model.add(ks.layers.Conv2D(64, 3, activation='relu'))
model.add(ks.layers.Conv2D(64, 3, activation='relu'))
model.add(ks.layers.Conv2D(64, 3, activation='relu'))
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(1024, activation='relu'))
model.add(ks.layers.Dense(10, activation='softmax'))

model.compile(ks.optimizers.Adam(0.001), loss=ks.losses.categorical_crossentropy)
model.fit(x_train, y_train, validation_data=(x_test, x_train))
