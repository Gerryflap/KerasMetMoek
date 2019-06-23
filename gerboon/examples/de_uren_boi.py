import numpy as np
import keras as ks
import matplotlib.pyplot as plt
import keras.backend as K

X = np.linspace(-1, 1, 24)
Y = np.concatenate([X[1:], X[:1]], axis=0)

X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)

X_sincos = np.concatenate([np.cos(X), np.sin(X)], axis=1)

model_1 = ks.Sequential()
model_1.add(ks.layers.Dense(100, activation='selu', input_shape=(1,)))
model_1.add(ks.layers.Dense(100, activation='selu'))
model_1.add(ks.layers.Dense(1, activation='tanh'))
model_1.compile(optimizer=ks.optimizers.Adam(0.0003), loss='mse')

model_2 = ks.Sequential()
model_2.add(ks.layers.Dense(100, activation='selu', input_shape=(2,)))
model_2.add(ks.layers.Dense(100, activation='selu'))
model_2.add(ks.layers.Dense(1, activation='tanh'))
model_2.compile(optimizer=ks.optimizers.Adam(0.0003), loss='mse')

model_3 = ks.Sequential()
model_3.add(ks.layers.Dense(2, input_shape=(1,)))
model_3.add(ks.layers.Lambda(lambda x: K.sin(x)))
model_3.add(ks.layers.Dense(100, activation='selu', input_shape=(1,)))
model_3.add(ks.layers.Dense(100, activation='selu'))
model_3.add(ks.layers.Dense(1, activation='tanh'))
model_3.compile(optimizer=ks.optimizers.Adam(0.0003), loss='mse')

hist1 = model_1.fit(X, Y, epochs=40000)
hist2 = model_2.fit(X_sincos, Y, epochs=40000)
hist3 = model_3.fit(X, Y, epochs=40000)


plt.plot(hist1.history['loss'][20000:], label="normal input")
plt.plot(hist2.history['loss'][20000:], label="sincos input")
plt.plot(hist3.history['loss'][20000:], label="learned sincos input")
plt.legend()
plt.show()

plt.plot(model_1.predict(X), label="normal pred")
plt.plot(model_2.predict(X_sincos), label="sincos pred")
plt.plot(model_3.predict(X), label="learned sincos pred")
plt.plot(Y, label="ground truth")
plt.legend()
plt.show()

