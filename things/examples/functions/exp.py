import keras as ks
import numpy as np
import matplotlib.pyplot as plt

model = ks.models.Sequential()
model.add(ks.layers.Dense(2500, input_dim=2, activation=ks.activations.relu))
model.add(ks.layers.Dense(1, activation=ks.activations.linear))

model.compile(loss=ks.losses.mean_squared_error, optimizer=ks.optimizers.Adam(lr=0.0005))

delta = 0.1
x = np.random.rand(4096, 1)*4 - 2
y = np.exp(x + 2 * delta)
x = np.exp(np.concatenate([x, x + delta], axis=1))

history = model.fit(x, y, epochs=30, batch_size=128)

prediction = np.array([[1.0, 1 + delta], [2.0, 2.0 + delta]])
print(model.predict(np.exp(prediction)))
print(np.exp([1.0 + 2 * delta, 2.0 + 2 * delta]))

exp_graph_x = np.reshape(np.arange(0, 5, 0.001), [-1,1])
exp_graph_true = np.exp(exp_graph_x)
exp_graph_nn_in = np.exp(np.concatenate([exp_graph_x-delta*2, exp_graph_x- delta], axis=1))
exp_graph_predict = model.predict(exp_graph_nn_in)
# plt.plot(np.log(history.history['loss']))
plt.plot(np.log(exp_graph_predict))
plt.plot(np.log(exp_graph_true))
plt.show()

