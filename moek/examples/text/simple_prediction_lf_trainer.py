import numpy as np
import keras as ks
from keras.callbacks import ModelCheckpoint

# load ascii text and covert to lowercase
file_path = "./sources/jules_verne.txt"
raw_text = open(file_path).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
alphabet = sorted(list(set(raw_text)))

def text_to_one_hot(text, alphabet=alphabet):
    one_hot_text = []
    for c in text:
        out = np.zeros(len(alphabet))
        out[alphabet.index(c)] = 1
        one_hot_text.append(out)
    one_hot_text = np.array(one_hot_text)
    return one_hot_text

def one_hot_to_text(one_hot, alphabet=alphabet):
    t = ""
    for sample in one_hot:
        pos = np.argmax(sample)
        t += alphabet[pos]
    return t

one_hot_text = text_to_one_hot(raw_text)
n_input_chars = 10

model = ks.models.Sequential()
model.add(ks.layers.LSTM(256, input_shape=(n_input_chars, len(alphabet)), return_sequences=True))
model.add(ks.layers.Dropout(0.2))
model.add(ks.layers.LSTM(256))
model.add(ks.layers.Dropout(0.2))
model.add(ks.layers.Dense(len(alphabet), activation=ks.activations.softmax))
model.compile(loss=ks.losses.categorical_crossentropy, optimizer=ks.optimizers.Adam(0.001))
# model.load_weights("weights-improvement-01-1.7923.hdf5")

# define the checkpoint
save_file_path="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(save_file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

t_dim = np.reshape(one_hot_text, [-1, 1, len(alphabet)])
# Generate 5 long sequences
x = np.concatenate([t_dim[i:len(t_dim)-(n_input_chars)+i] for i in range(n_input_chars)], axis=1)
y = np.reshape(t_dim[n_input_chars:], [-1, len(alphabet)])

try:
    history = model.fit(x, y, epochs=2, batch_size=1024, callbacks=callbacks_list)
except KeyboardInterrupt:
    print("Training was interrupted")

