import keras as ks
import numpy as np
import matplotlib.pyplot as plt

model = ks.models.Sequential()
model.add(ks.layers.Dense(16, input_dim=1, activation=ks.activations.relu))
model.add(ks.layers.Dense(500, activation=ks.activations.relu))
model.add(ks.layers.Dense(100, activation=ks.activations.elu))
model.add(ks.layers.Dense(1, activation=ks.activations.linear))

model.compile(loss=ks.losses.mean_squared_error, optimizer=ks.optimizers.Adam(lr=0.000025))

x = np.random.rand(4096)*100
y = x

history = model.fit(x, y, epochs=250, batch_size=256)

plt.plot(np.log(history.history['loss']))
plt.show()

prediction = np.array([1, 5, 10, 20, 100, 1000000])
print(model.predict(prediction))
print(prediction)
