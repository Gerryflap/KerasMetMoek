import time

from util.math import unison_shuffled_copies
from util.text import filter_text, text_to_one_hot
import keras as ks
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import os

load_model = input("Load model? (y/N)") == "y"
word_length = 20


def split_pad_text(text):
    text = text.split(" ")
    for i in range(len(text)):
        if len(text[i]) > word_length:
            text[i] = text[i][:word_length]
        elif len(text[i]) < word_length:
            text[i] += " " * (word_length - len(text[i]))

    return text


with open("../../../sources/books/jules_verne.txt", "r") as f:
    dutch_words = filter_text(f.read())
    d_alphabet = set(dutch_words)
    print("d_alphabet: " , sorted(list(d_alphabet)))
    dutch_words = split_pad_text(dutch_words)

with open("../../../sources/books/verwandlung", "r") as f:
    german_words = filter_text(f.read())
    g_alphabet = set(german_words)
    print("g_alphabet: " , sorted(list(g_alphabet)))
    german_words = split_pad_text(german_words)

alphabet = sorted(list(g_alphabet | d_alphabet))

print(alphabet)

x = np.array([text_to_one_hot(word, alphabet) for word in german_words + dutch_words])
dutch_y = np.concatenate([np.zeros([len(dutch_words), 1]), np.zeros([len(dutch_words), 1]) + 1], axis=1)
german_y = np.concatenate([np.zeros([len(german_words), 1]) + 1, np.zeros([len(german_words), 1])], axis=1)
print(dutch_y.shape, german_y.shape)
y = np.concatenate([german_y, dutch_y], axis=0)

x, y = unison_shuffled_copies(x, y)

if not load_model:
    model = ks.models.Sequential()
    model.add(ks.layers.LSTM(32, input_shape=(word_length, len(alphabet))))
    model.add(ks.layers.Dense(32, activation=ks.activations.tanh))
    model.add(ks.layers.Dense(2, activation=ks.activations.softmax))
    model.compile(optimizer=ks.optimizers.Adam(lr=0.003), loss=ks.losses.categorical_crossentropy)
else:
    files = os.listdir(".")
    print("Which model do you want?")
    files.sort()
    files = list(filter(lambda path: "best_model" in path, files))
    print("id\tfilename")
    for i in range(len(files)):
        print(str(i) + "\t" + files[i])
    n = int(input("file id: "))
    print("Loading model")
    model = ks.models.load_model(files[n])

save_file_path = "best_model_%d.hdf5" % time.time()
checkpoint = ModelCheckpoint(save_file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(x, y, epochs=100, batch_size=512, callbacks=callbacks_list)

plt.plot(np.log(history.history['loss']))
plt.show()

while True:
    input_word = input("Give a word (or multiple words split with spaces): ")
    input_words = split_pad_text(input_word)
    x_input = np.array([text_to_one_hot(word, alphabet) for word in input_words])
    print(model.predict(x_input))
