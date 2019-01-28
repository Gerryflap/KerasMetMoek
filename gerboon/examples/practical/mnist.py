import keras as ks
from keras.datasets import mnist

# Load mnist:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Encode the outputs as one-hot vectors (example: class 2 becomes [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y_train, y_test = ks.utils.to_categorical(y_train), ks.utils.to_categorical(y_test)

model = ks.models.Sequential()
# Flatten images to one dimension
model.add(ks.layers.Flatten(input_shape=(28, 28,)))

# First layer
model.add(ks.layers.Dense(256, activation='tanh'))

# Second layer (otherwise it would not be deep learning and therefore lame)
model.add(ks.layers.Dense(128, activation='tanh'))


# Output layer (softmax activation gives a probability distribution as output)
model.add(ks.layers.Dense(10, activation='softmax'))

# Give the model a loss function to optimize, an optimizer to do so and give it extra metrics (accuracy) that will be reported
model.compile(optimizer=ks.optimizers.Adam(lr=0.00005), loss=ks.losses.categorical_crossentropy, metrics=['acc'])

# Fit the model to the data (run 10 times over the dataset) :
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
