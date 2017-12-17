import keras as ks
import numpy as np
from keras.datasets import boston_housing
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Normalize input
x_train, x_test = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0), (x_test - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
y_train, y_test = y_train/10, y_test/10
print("Data information:")
print("Mean: ", np.mean(x_train, axis=0))
print("Variance: ", np.var(x_train, axis=0))

# Initialize the models
models = {}

# Model 1, wide 13x10000 + 10000 = 140000 weights
model = ks.models.Sequential()
model.add(ks.layers.Dense(10000, input_shape=(13,), activation='elu'))
model.add(ks.layers.Dense(1, activation='linear'))

models["wide"] = model

# Model 2, deep, 8715 weights:
model = ks.models.Sequential()
model.add(ks.layers.Dense(100, input_shape=(13,), activation='elu'))
model.add(ks.layers.Dense(50,  activation='elu'))
model.add(ks.layers.Dense(30,  activation='elu'))
model.add(ks.layers.Dense(20,  activation='elu'))
model.add(ks.layers.Dense(15,  activation='elu'))
model.add(ks.layers.Dense(1, activation='linear'))

models["deep"] = model

# Model 3, small, 13x20 + 20*10 + 10 = 470 weights
model = ks.models.Sequential()
model.add(ks.layers.Dense(20, input_shape=(13,), activation='elu'))
model.add(ks.layers.Dense(10,  activation='elu'))
model.add(ks.layers.Dense(1, activation='linear'))

models["small"] = model

# Model 4, very_deep, veel weights, MEER LAYERS:
model = ks.models.Sequential()
model.add(ks.layers.Dense(300, input_shape=(13,), activation='elu'))
model.add(ks.layers.Dense(200,  activation='elu'))
model.add(ks.layers.Dense(100,  activation='elu'))
model.add(ks.layers.Dense(100,  activation='elu'))
model.add(ks.layers.Dense(100,  activation='elu'))
model.add(ks.layers.Dense(100,  activation='elu'))
model.add(ks.layers.Dense(100,  activation='elu'))
model.add(ks.layers.Dense(100,  activation='elu'))
model.add(ks.layers.Dense(50,  activation='elu'))
model.add(ks.layers.Dense(50,  activation='elu'))
model.add(ks.layers.Dense(30,  activation='elu'))
model.add(ks.layers.Dense(1, activation='linear'))

models["very_deep"] = model

performance = {}
training_mse = {}
for name, model in models.items():
    # Compile the model
    model.compile(optimizer=ks.optimizers.Adam(0.0001), loss=ks.losses.mean_squared_error, metrics=[ks.losses.mean_absolute_error])

    print("Training model ", name)
    h = model.fit(x_train, y_train, epochs=100)
    print("Training done for model ", name)

    training_mse[name] = h.history['loss']
    performance[name] = model.evaluate(x_test, y_test)


print("Results:")
for name, perf in performance.items():
    print("Model performance of %s: " % name, perf)

for name, history in training_mse.items():
    plt.plot(history, label=name)
plt.legend()
plt.show()
