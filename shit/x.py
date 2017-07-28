import keras as ks
import numpy as np
import math

model = ks.models.Sequential()
model.add(ks.layers.Dense(16, input_dim=1, activation=ks.activations.relu))
model.add(ks.layers.Dense(1, activation=ks.activations.linear))

model.compile(loss=ks.losses.mean_squared_error, optimizer=ks.optimizers.Adam(0.05))

x = np.random.rand(512)*10
y = x

model.fit(x, y, epochs=10, batch_size=1)

prediction = np.array([1, 5, 10, 20, 100, 1000000])
print(model.predict(prediction))
print(np.sin(prediction))
