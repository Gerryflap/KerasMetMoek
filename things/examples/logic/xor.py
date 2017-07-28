import keras as ks
import numpy as np

model = ks.models.Sequential()
model.add(ks.layers.Dense(4, input_dim=2, activation=ks.activations.tanh))
model.add(ks.layers.Dense(1, activation=ks.activations.linear))

model.compile(loss=ks.losses.mean_squared_error, optimizer=ks.optimizers.Adam(0.01))

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model.fit(x, y, epochs=1000, batch_size=4)

print(model.predict(x))
