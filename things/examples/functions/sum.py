import keras as ks
import numpy as np
import matplotlib.pyplot as plt


model = ks.models.Sequential()
model.add(ks.layers.Dense(10, input_shape=(25, 1), activation='tanh'))
model.add(ks.layers.LSTM(120, return_sequences=True))
model.add(ks.layers.LSTM(120))
model.add(ks.layers.Dense(1, activation=ks.activations.linear))

model.compile(loss=ks.losses.mean_squared_error, optimizer=ks.optimizers.Adam(0.01))

x = np.random.rand(200, 25, 1)
y = np.sum(x, axis=1)

history = model.fit(x, y, epochs=50, batch_size=128)

print(x[0])
print(model.predict(x[0:1]))
print(y[0])
plt.plot(np.log(history.history['loss']))
plt.show()


