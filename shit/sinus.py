import keras as ks
import numpy as np
import math

model = ks.models.Sequential()
model.add(ks.layers.Dense(15, input_dim=1, activation=ks.activations.tanh))
model.add(ks.layers.Dense(1, activation=ks.activations.tanh))

model.compile(loss=ks.losses.mean_squared_error, optimizer=ks.optimizers.Adam(0.1))

x = np.random.rand(4096)*math.pi*2
y = np.sin(x)

model.fit(x, y, epochs=10, batch_size=8)

print(model.predict(np.array([0, math.pi, math.pi*2])))