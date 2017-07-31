import codecs

import keras as ks
import numpy as np

from util.math import unison_shuffled_copies
from util.text import filter_text, text_to_one_hot, one_hot_to_text

word_length = 20


def split_pad_text(text):
    text = text.split(" ")
    for i in range(len(text)):
        if len(text[i]) > word_length:
            text[i] = text[i][:word_length]
        elif len(text[i]) < word_length:
            text[i] += " " * (word_length - len(text[i]))

    return text


with codecs.open("../../../sources/bier/nederlands_bier", "r", encoding='utf-8') as f:
    dutch_words = filter_text(f.read())
    alphabet = set(dutch_words)
    dutch_words = split_pad_text(dutch_words)
    dutch_words = [word for word in dutch_words if word[0] != " "]

alphabet = sorted(list(alphabet))

words_one_hot = np.array([text_to_one_hot(word, alphabet) for word in dutch_words])
print(alphabet)


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

# Build Generator
gen_opt = ks.optimizers.Adam(0.0005)
gen_model = ks.models.Sequential()
gen_model.add(ks.layers.LSTM(64, input_shape=(word_length, 5), return_sequences=True))
gen_model.add(ks.layers.LSTM(32, return_sequences=True))
gen_model.add(ks.layers.Dropout(0.5))
gen_model.add(ks.layers.Dense(len(alphabet), activation=ks.activations.tanh))
gen_model.compile(optimizer=gen_opt, loss=ks.losses.categorical_crossentropy)

# Build Discriminator
discr_model = ks.models.Sequential()
discr_model.add(ks.layers.LSTM(64, input_shape=(word_length, len(alphabet)), return_sequences=True))
discr_model.add(ks.layers.LSTM(32, return_sequences=True))
discr_model.add(ks.layers.LSTM(32))
discr_model.add(ks.layers.Dense(2, activation=ks.activations.softmax))
discr_model.compile(optimizer=ks.optimizers.SGD(lr=0.002), loss=ks.losses.categorical_crossentropy)

# Build stacked model
make_trainable(discr_model, False)
gan = ks.models.Sequential()
gan.add(gen_model)
gan.add(discr_model)
gan.compile(optimizer=gen_opt, loss=ks.losses.categorical_crossentropy)

epoch_size = 2000

for i in range(1000):
    # generate words
    noise = np.random.rand(epoch_size // 2, word_length, 5)
    words = gen_model.predict(noise)
    #print(words)

    # Train discriminator
    np.random.shuffle(words_one_hot)
    x = np.concatenate([words_one_hot[:epoch_size // 2], words], axis=0)
    y = np.zeros([epoch_size, 2])
    y[:epoch_size // 2, 0] = 1
    y[epoch_size // 2:, 1] = 1

    # unified shuffle
    x, y = unison_shuffled_copies(x, y)

    print("Training discriminative model")
    discr_model.fit(x, y, epochs=1, batch_size=200)

    # Train stack
    noise = np.random.rand(epoch_size, word_length, 5)
    y = np.zeros([epoch_size, 2])
    y[:, 0] = 1
    print("Training stack")
    gan.fit(noise, y, epochs=1, batch_size=200)

    if i%10 == 0:
        for one_hot_word in gen_model.predict(np.random.rand(10,word_length,5)):
            print(one_hot_to_text(one_hot_word, alphabet))
        print(discr_model.predict(gen_model.predict(np.random.rand(10,word_length,5))))