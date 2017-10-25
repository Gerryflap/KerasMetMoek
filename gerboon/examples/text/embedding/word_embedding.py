import codecs
import numpy as np
import keras as ks
import random
import matplotlib.pyplot as plt
from scipy import spatial

alphabet = " abcdefghijklmnopqrstuvwxyz"
alphabet_index = dict()
for i, char in enumerate(alphabet):
    alphabet_index[char] = i

word_length = 10

with codecs.open("../../../../sources/books/jules_verne.txt", "r", encoding='utf-8') as f:
    data = f.read()

#data = "Het is natuurlijk zeker waar dat bananen lekker zijn. maar wist u dat banananen naast lekker, ook nog eens heel gezond zijn?"

data = data.lower()

def word_to_one_hot(word):
    word_one_hot = []
    for char in word:
        char_one_hot = np.zeros((len(alphabet),))
        char_one_hot[alphabet_index.get(char, 0)] = 1
        word_one_hot.append(char_one_hot)
    return np.array(word_one_hot)


def generate_one_hots(text, target_window=2, negative_targets=10):
    """
    This function converts a whole text collection to one-hot encoded word, target word, and negative target structures
    :param text: Input text
    :param target_window: Size of context window around words
    :return: words one hot, context per word one hot and negative targets per word one hot
    """
    words = text.split()
    output = np.zeros((len(words), word_length, len(alphabet)))
    target_output = np.zeros((len(words), target_window*2, word_length, len(alphabet)))
    n_target_output = np.zeros((len(words), negative_targets, word_length, len(alphabet)))
    text_output = []
    output[:, :, 0] = 1
    target_output[:, :, :, 0] = 1
    for i in range(target_window, len(words)-target_window):
        target_words = words[i - target_window:i + target_window + 1]
        tw_set = set(target_words)
        neg_set = set()
        while len(neg_set) != negative_targets:
            w = words[random.randint(0,len(words)-1)]
            if w in tw_set or w in neg_set:
                continue
            else:
                neg_set.add(w)
        word = words[i]
        w_oh = word_to_one_hot(word)[:word_length]
        if len(w_oh) == 0:
            continue
        output[i, :len(w_oh)] = w_oh
        text_output.append(word[:word_length])

        # print("Word: ", word)
        # print("Targets: ", tw_set)
        # print("Negative: ", neg_set)

        j = 0
        for w in target_words:
            if w == word:
                continue
            tw_oh = word_to_one_hot(w)[:word_length]
            if len(tw_oh) == 0:
                continue
            target_output[i,j,:len(tw_oh)] = tw_oh
            j += 1
        j = 0
        for w in neg_set:
            ne_oh = word_to_one_hot(w)[:word_length]
            if len(ne_oh) == 0:
                continue
            n_target_output[i, j, :len(ne_oh)] = ne_oh

    return output[target_window: -target_window], \
           target_output[target_window: -target_window], \
           n_target_output[target_window: -target_window], \
           text_output


def batch_generator(words_oh, targets_oh, negs_oh, batch_size=10):
    while True:
        batch_words = []
        batch_targets = []
        batch_ys = []
        for i in range(words_oh.shape[0]):
            if random.random() > 0.5:
                target = targets_oh[i, random.randint(0, targets_oh.shape[1]-1)]
                y = 1
            else:
                target = negs_oh[i, random.randint(0, negs_oh.shape[1]-1)]
                y = 0
            word = words_oh[i]
            batch_words.append(word)
            batch_targets.append(target)
            batch_ys.append(y)
            if len(batch_words) == batch_size:
                yield [np.array(batch_words), np.array(batch_targets)], np.array(batch_ys)


# The model:
input_word = ks.Input((word_length, len(alphabet)))
input_target = ks.Input((word_length, len(alphabet)))

word_to_vec = ks.layers.LSTM(10, return_sequences=False)

word_vec = word_to_vec(input_word)
target_vec = word_to_vec(input_target)

total_vec = ks.layers.Concatenate(axis=1)([word_vec, target_vec])
is_target = ks.layers.Dense(1, activation=ks.activations.sigmoid)(total_vec)

model = ks.models.Model(inputs=[input_word, input_target], outputs=is_target)
model.compile(ks.optimizers.Adam(0.001), ks.losses.binary_crossentropy)

vector_predictor = ks.models.Model(inputs=input_word, outputs=word_vec)
vector_predictor.compile(ks.optimizers.Adam(0.001), ks.losses.binary_crossentropy)
# End model definition


words_oh, targets_oh, negs_oh, words = generate_one_hots(data, target_window=5, negative_targets=10)
#words_oh, targets_oh, negs_oh, words = generate_one_hots(data, target_window=2, negative_targets=4)
#print(np.argmax(words_oh, axis=2))
model.fit_generator(batch_generator(words_oh, targets_oh, negs_oh, batch_size=2), steps_per_epoch=30, epochs=10)

preds = vector_predictor.predict(words_oh)

while True:
    word = input("Give a word: ")
    word_oh = np.zeros((1, word_length, len(alphabet)))
    word_oh[:, :, 0] = 1
    oh_v = word_to_one_hot(word.lower())
    word_oh[0, :len(oh_v)] = oh_v
    print(word_oh)

    vector = vector_predictor.predict(word_oh)
    costs = np.array([spatial.distance.cosine(vector[0], v) for v in preds])
    sort = np.argsort(costs)
    closest_words = [words[i] for i in sort]
    print(closest_words)
