import numpy as np
import random
import keras as ks


def add_coordinates(frame):
    x = np.linspace(-1, 1, frame.shape[0])
    y = np.linspace(-1, 1, frame.shape[1])
    xv, yv = np.meshgrid(x, y)
    coords = np.stack([xv, yv], axis=2)
    frame = np.concatenate([frame, coords], axis=2)
    return frame


def gen_samples(batch_size, size, f=lambda x: x):
    while True:
        Xs = []
        Ys = []
        for i in range(batch_size):
            X = np.zeros((size, size, 1))
            Y = np.zeros((2,))
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            X[y, x] = 1
            Y[0] = x
            Y[1] = y
            X = f(X)
            Xs.append(X)
            Ys.append(Y)
        Xs = np.stack(Xs, axis=0)
        Ys = np.stack(Ys, axis=0)
        yield Xs, Ys


size = 10

print(add_coordinates(np.zeros((size, size, 1))))

model1 = ks.models.Sequential()
model1.add(ks.layers.Conv2D(1, 3, strides=2, input_shape=(size, size, 1), activation='selu', padding='same'))
model1.add(ks.layers.Conv2D(1, 3, strides=2, activation='selu', padding='same'))
model1.add(ks.layers.Flatten())
model1.add(ks.layers.Dense(2, activation='linear'))
model1.compile(optimizer='adam', loss='mse')


model2 = ks.models.Sequential()
model2.add(ks.layers.Conv2D(1, 3, strides=2, input_shape=(size, size, 3), activation='selu', padding='same'))
model2.add(ks.layers.Conv2D(1, 3, strides=2, activation='selu', padding='same'))
model2.add(ks.layers.Flatten())
model2.add(ks.layers.Dense(2, activation='linear'))
model2.compile(optimizer='adam', loss='mse')

model2.summary()

model1.fit_generator(gen_samples(32, size), steps_per_epoch=1000, epochs=5)
model2.fit_generator(gen_samples(32, size, add_coordinates), steps_per_epoch=1000, epochs=5)