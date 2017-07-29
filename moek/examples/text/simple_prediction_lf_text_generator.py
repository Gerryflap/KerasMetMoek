import keras as ks
import numpy as np


def text_to_one_hot(text, alphabet):
    one_hot_text = []
    for c in text:
        out = np.zeros(len(alphabet))
        out[alphabet.index(c)] = 1
        one_hot_text.append(out)
    one_hot_text = np.array(one_hot_text)
    return one_hot_text

def one_hot_to_text(one_hot, alphabet):
    t = ""
    for sample in one_hot:
        pos = np.argmax(sample)
        t += alphabet[pos]
    return t

def alphabet_frequency(string, alphabet):
    result = {}
    for char in string:
        result[char] = result.get(char, 0) + 1
    result = list(result.items())
    return sorted(result, key=lambda elem: -elem[1])


# load ascii text and covert to lowercase
file_path = "./sources/jules_verne.txt"
raw_text = open(file_path).read()
raw_text = raw_text.lower()

alphabet = ""
for c in raw_text:
    if c not in alphabet:
        alphabet += c

n_input_chars = 10

# load the network weights
filename = "weights-improvement-01-2.1400.hdf5"
model = ks.models.load_model(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(alphabet)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

one_hot_text = text_to_one_hot(raw_text, alphabet)

t_dim = np.reshape(one_hot_text, [-1, 1, len(alphabet)])
# Generate 5 long sequences
x = np.concatenate([t_dim[i:len(t_dim)-(n_input_chars)+i] for i in range(n_input_chars)], axis=1)
y = np.reshape(t_dim[n_input_chars:], [-1, len(alphabet)])

predicted_text = np.reshape(model.predict(x), [-1, len(alphabet)])

print(one_hot_to_text(predicted_text, alphabet))
