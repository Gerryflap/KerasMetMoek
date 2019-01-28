import codecs

import keras as ks
import numpy as np

from util.math import unison_shuffled_copies
from util.text import filter_text, text_to_one_hot, one_hot_to_text

word_length = 8


def pad_text(text):
    for i in range(len(text)):
        if len(text[i]) == word_length:
            text[i] = text[i][:word_length]
        elif len(text[i]) < word_length:
            text[i] += " " * (word_length - len(text[i]))

    return text


with codecs.open("../../../sources/other/top1000_nl.txt", "r", encoding='utf-8') as f:
    dutch_words = filter_text(f.read())
    d_alphabet = set(dutch_words)
    dutch_words = dutch_words.split(" ")
    dutch_words = pad_text(list({word for word in dutch_words if len(word) < word_length}))
    dutch_words = [word for word in dutch_words if word[0] != " "]

with codecs.open("../../../sources/other/top1000_ger.txt", "r", encoding='utf-8') as f:
    german_words = filter_text(f.read())
    g_alphabet = set(german_words)
    german_words = german_words.split(" ")
    print("g_alphabet: " , sorted(list(g_alphabet)))
    german_words = pad_text(list({word for word in german_words if len(word) < word_length}))
    german_words = [word for word in german_words if word[0] != " "]

print("Length of German words: ", len(german_words))
print("Length of Dutch words: ", len(dutch_words))

alphabet = sorted(list(d_alphabet | g_alphabet))

words_one_hot_dutch = np.array([text_to_one_hot(word, alphabet) for word in dutch_words])
words_one_hot_german = np.array([text_to_one_hot(word, alphabet) for word in german_words])
print(alphabet)


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def build_single_rnn_generator_model(name):

    inp = ks.Input(shape=(word_length, len(alphabet)))
    x = ks.layers.Bidirectional(ks.layers.GRU(256, return_sequences=True), input_shape=(word_length, len(alphabet)))(inp)
    # For a bottleneck: x = ks.layers.RepeatVector(word_length,)(x)
    # x = ks.layers.Bidirectional(ks.layers.GRU(256, return_sequences=True))(x)
    x = ks.layers.Concatenate(axis=2)([inp, x])

    out =  ks.layers.Bidirectional(ks.layers.GRU(len(alphabet), activation='softmax', return_sequences=True), merge_mode='sum')(x)
    # out = ks.layers.TimeDistributed(ks.layers.Dense(len(alphabet), activation='softmax'))(x)
    model = ks.Model(inputs=inp, outputs=out, name=name)
    return model


def build_single_rnn_discriminator_model(name):
    reg = ks.regularizers.l1_l2(0.000001, 0.000001)
    model = ks.models.Sequential(name=name)
    model.add(ks.layers.Bidirectional(ks.layers.GRU(128, kernel_regularizer=reg, return_sequences=False), input_shape=(word_length, len(alphabet))))
    #model.add(ks.layers.Bidirectional(ks.layers.GRU(32, kernel_regularizer=reg, return_sequences=False)))
    model.add(ks.layers.Dense(1, activation='sigmoid'))
    return model

def build_single_cnn_generator_model(name):

    model = ks.models.Sequential(name=name)
    model.add(ks.layers.Conv1D(16, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(16, 3, strides=2, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(32, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'))
    # Upsample back:

    model.add(ks.layers.Conv1D(64, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(64, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.UpSampling1D(2))
    model.add(ks.layers.Conv1D(32, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(32, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.UpSampling1D(2))
    model.add(ks.layers.Conv1D(16, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(len(alphabet), 3, strides=1, padding='same', activation='softmax'))
    return model


def build_single_cnn_discriminator_model(name):
    model = ks.models.Sequential(name=name)
    model.add(ks.layers.Conv1D(16, 3, strides=1, padding='same', activation='relu', input_shape=(word_length, len(alphabet))))
    model.add(ks.layers.Conv1D(16, 3, strides=2, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(32, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(64, 3, strides=1, padding='same', activation='relu'))
    model.add(ks.layers.Conv1D(64, 3, strides=2, padding='same', activation='relu'))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(1, activation='sigmoid'))
    return model

def build_double_discriminator_model(name):
    inp = ks.Input(shape=(word_length, len(alphabet)))

    cnn_model = ks.models.Sequential(name=name)
    cnn_model.add(ks.layers.Conv1D(16, 3, strides=1, padding='same', activation='relu', input_shape=(word_length, len(alphabet))))
    cnn_model.add(ks.layers.Conv1D(16, 3, strides=2, padding='same', activation='relu'))
    cnn_model.add(ks.layers.Conv1D(32, 3, strides=1, padding='same', activation='relu'))
    cnn_model.add(ks.layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'))
    cnn_model.add(ks.layers.Conv1D(64, 3, strides=1, padding='same', activation='relu'))
    cnn_model.add(ks.layers.Conv1D(64, 3, strides=2, padding='same', activation='relu'))
    cnn_model.add(ks.layers.Flatten())
    cnn_model.add(ks.layers.Dense(10, activation='sigmoid'))

    cnn_out = cnn_model(inp)
    rnn_out = ks.layers.Bidirectional(ks.layers.GRU(128, return_sequences=False))(inp)
    rnn_out = ks.layers.Dense(10, activation='sigmoid')(rnn_out)

    merge = ks.layers.Concatenate(axis=1)([cnn_out, rnn_out])
    out = ks.layers.Dense(1, activation='sigmoid')(merge)
    model = ks.models.Model(inputs=inp, outputs=out, name=name)
    return model

def build_generator_loss(discriminator):
    def generator_loss(y_pred, y_target):
        return ks.losses.binary_crossentropy(discriminator(y_pred), y_target)
    return generator_loss


def sample_random(array, size):
    return array[np.random.randint(0, array.shape[0], size)]


G_dutch_german = build_single_rnn_generator_model("G_dg")
G_german_dutch = build_single_rnn_generator_model("G_gd")
D_dutch = build_double_discriminator_model("D_d")
D_german = build_double_discriminator_model("D_g")

D_dutch.compile(ks.optimizers.Adam(0.003), loss=ks.losses.binary_crossentropy)
D_german.compile(ks.optimizers.Adam(0.003), loss=ks.losses.binary_crossentropy)

# Define Dutch -> German -> Dutch cycle model
dutch_input = ks.Input(shape=(word_length, len(alphabet)))
generated_german = G_dutch_german(dutch_input)
make_trainable(D_german, False)
discriminator_german_pred = D_german(generated_german)
dutch_cycle_out = G_german_dutch(generated_german)

dutch_cycle_model = ks.Model(inputs=dutch_input, outputs=[discriminator_german_pred, dutch_cycle_out])
dutch_cycle_model.compile(ks.optimizers.Adam(0.003), loss=[ks.losses.binary_crossentropy, ks.losses.categorical_crossentropy])

# Define German -> Dutch -> German cycle model
german_input = ks.Input(shape=(word_length, len(alphabet)))
generated_dutch = G_german_dutch(german_input)
make_trainable(D_dutch, False)
discriminator_dutch_pred = D_dutch(generated_dutch)
german_cycle_out = G_dutch_german(generated_dutch)

german_cycle_model = ks.Model(inputs=german_input, outputs=[discriminator_dutch_pred, german_cycle_out])
german_cycle_model.compile(ks.optimizers.Adam(0.003), loss=[ks.losses.binary_crossentropy, ks.losses.categorical_crossentropy])

# Training Loop:
epochs = 100000
batch_size = 32

G_batches_per_epoch = 4
D_batches_per_epoch = 1

real_dutch_fixed = sample_random(words_one_hot_dutch, 5)
real_german_fixed = sample_random(words_one_hot_german, 5)

for i in range(epochs):

    # Update Discriminators
    for j in range(D_batches_per_epoch):
        real_dutch = sample_random(words_one_hot_dutch, batch_size//2)
        real_german = sample_random(words_one_hot_german, batch_size//2)

        fake_dutch = G_german_dutch.predict(real_german)
        fake_german = G_dutch_german.predict(real_dutch)

        D_d_labels = np.zeros((batch_size, 1))
        D_d_labels[batch_size//2:] = 1

        D_g_labels = np.zeros((batch_size, 1))
        D_g_labels[batch_size//2:] = 1

        D_d_inputs = np.concatenate([fake_dutch, real_dutch], axis=0)
        D_g_inputs = np.concatenate([fake_german, real_german], axis=0)

        h = D_dutch.fit(D_d_inputs, D_d_labels, batch_size=batch_size, verbose=False)
        D_dutch_loss = h.history['loss'][0]
        h = D_german.fit(D_g_inputs, D_g_labels, batch_size=batch_size, verbose=False)
        D_german_loss = h.history['loss'][0]


    # Update Generators
    for j in range(G_batches_per_epoch):
        real_dutch = sample_random(words_one_hot_dutch, batch_size)
        real_german = sample_random(words_one_hot_german, batch_size)

        labels = np.ones((batch_size, 1))

        h = dutch_cycle_model.fit(real_dutch, [labels, real_dutch], batch_size=batch_size, verbose=False)
        dutch_cycle_loss = h.history['G_gd_loss'][0]
        german_gen_loss = h.history['D_g_loss'][0]
        h = german_cycle_model.fit(real_german, [labels, real_german], batch_size=batch_size, verbose=False)
        german_cycle_loss = h.history['G_dg_loss'][0]
        dutch_gen_loss = h.history['D_d_loss'][0]

    # Show results
    if i%10 == 0:

        predicted_german = G_dutch_german.predict(real_dutch_fixed)
        predicted_dutch = G_german_dutch.predict(real_german_fixed)

        print("Real German" , "->",  "Predicted Dutch")

        for oh_g, oh_d in zip(real_german_fixed, predicted_dutch):
            print(one_hot_to_text(oh_g, alphabet), "->", one_hot_to_text(oh_d, alphabet))

        print()

        print("Real Dutch", "->", "Predicted German")

        for oh_d, oh_g in zip(real_dutch_fixed, predicted_german):
            print(one_hot_to_text(oh_d, alphabet), "->", one_hot_to_text(oh_g, alphabet))

        print()

        print("Real German" , "->",  "Cycle German")

        cycle_german = german_cycle_model.predict(real_german_fixed)[1]

        for oh_g, oh_gc in zip(real_german_fixed, cycle_german):
            print(one_hot_to_text(oh_g, alphabet), "->", one_hot_to_text(oh_gc, alphabet))

        print()

        print("Real Dutch" , "->",  "Cycle Dutch")

        cycle_dutch = dutch_cycle_model.predict(real_dutch_fixed)[1]

        for oh_d, oh_dc in zip(real_dutch_fixed, cycle_dutch):
            print(one_hot_to_text(oh_d, alphabet), "->", one_hot_to_text(oh_dc, alphabet))

        print()

        print("Episode %d:"%i)
        print("Discriminator losses: Dutch = %.3f \t German = %.3f"%(D_dutch_loss, D_german_loss))
        print("D->G->D losses: Gen = %.3f \t Cycle = %.3f"%(german_gen_loss, dutch_cycle_loss))
        print("G->D->G losses: Gen = %.3f \t Cycle = %.3f"%(dutch_gen_loss, german_cycle_loss))
        print()