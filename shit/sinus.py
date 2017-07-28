import keras as ks
import numpy as np
import math

model = ks.models.Sequential()
model.add(ks.layers.Dense(2, input_dim=1, activation=ks.activations.tanh))
model.add(ks.layers.Dense(1, activation=ks.activations.linear))

model.compile(loss=ks.losses.mean_squared_error, optimizer=ks.optimizers.Adam(0.1))

x = np.random.rand(1024)*math.pi*2
y = np.sin(x)

model.fit(x, y, epochs=2, batch_size=1)

prediction = np.array([0, math.pi * 0.5, math.pi * 2])
print(model.predict(prediction))
print(np.sin(prediction))