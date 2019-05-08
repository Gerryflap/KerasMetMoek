import codecs

import keras as ks
import numpy as np

from util.math import unison_shuffled_copies
from util.text import filter_text, text_to_one_hot, one_hot_to_text

word_length = 5
inputs = 40


def split_pad_text(text):
    text = text.split(" ")
    for i in range(len(text)):
        if len(text[i]) > word_length:
            text[i] = text[i][:word_length]
        elif len(text[i]) < word_length:
            text[i] += " " * (word_length - len(text[i]))

    return text


with codecs.open("../../../sources/other/kernfusie.txt", "r", encoding='utf-8') as f:
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
gen_opt = ks.optimizers.Adam(0.00002)
gen_model = ks.models.Sequential()
gen_model.add(ks.layers.CuDNNGRU(128, input_shape=(word_length, inputs), return_sequences=True))
#gen_model.add(ks.layers.CuDNNGRU(128, return_sequences=True))
#gen_model.add(ks.layers.Dropout(0.5))
gen_model.add(ks.layers.CuDNNGRU(len(alphabet), return_sequences=True))
#gen_model.add(ks.layers.Dropout(0.5))
gen_model.compile(optimizer=gen_opt, loss=ks.losses.binary_crossentropy)

# Build Discriminator
discr_model = ks.models.Sequential()
discr_model.add(ks.layers.CuDNNGRU(128, input_shape=(word_length, len(alphabet)), return_sequences=True))
discr_model.add(ks.layers.CuDNNGRU(128))
discr_model.add(ks.layers.Dense(1, activation=ks.activations.sigmoid))
discr_model.compile(optimizer=ks.optimizers.Adam(lr=0.00001), loss=ks.losses.binary_crossentropy)

# Build stacked model
make_trainable(discr_model, False)
gan = ks.models.Sequential()
gan.add(gen_model)
gan.add(discr_model)
gan.compile(optimizer=gen_opt, loss=ks.losses.binary_crossentropy)


# Train discriminative model first
discr_y = np.zeros([len(words_one_hot)*2, 1])
discr_y[:len(words_one_hot), 0] = 1.0
discr_y[len(words_one_hot):, 0] = 0.0
discr_x = np.concatenate([words_one_hot, np.random.rand(len(words_one_hot), word_length, len(alphabet))])
discr_model.fit(discr_x, discr_y, epochs=10, batch_size=256)

epoch_size = 200

for i in range(10000):
    # generate words
    noise = np.random.standard_normal([epoch_size // 2, word_length, inputs])
    words = gen_model.predict(noise)
    #print(words)

    # Train discriminator
    np.random.shuffle(words_one_hot)
    x = np.concatenate([words_one_hot[:epoch_size // 2], words], axis=0)
    y = np.zeros([epoch_size, 1])
    y[:epoch_size // 2, 0] = 1.0
    y[epoch_size // 2:, 0] = 0.0

    # unified shuffle
    x, y = unison_shuffled_copies(x, y)

    print("Training discriminative model")
    discr_model.fit(x, y, epochs=1, batch_size=20, verbose=True)

    # Train stack
    noise = np.random.standard_normal([epoch_size, word_length, inputs])
    y = np.zeros([epoch_size, 1])
    y[:, 0] = 1.0
    print("Training stack")
    gan.fit(noise, y, epochs=1, batch_size=20, verbose=True)

    if i%10 == 0:
        pred = gen_model.predict(np.random.standard_normal([10,word_length,inputs]))
        for one_hot_word in pred:
            print(one_hot_to_text(one_hot_word, alphabet).replace(" ", "_"))
        print(discr_model.predict(pred))