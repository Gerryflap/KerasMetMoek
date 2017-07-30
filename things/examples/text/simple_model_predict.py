import keras as ks
import numpy as np
import matplotlib.pyplot as plt

n_input_chars = 15
#text = "abababababababababababababab"
with open('text', 'r') as f:
    text_original = f.read()
text_original = text_original.lower()
text_original = list(filter(lambda c: c in "abcdefghijklmnopqrstuvwxyz1234567890 ", text_original))

alphabet = ""
for c in text_original:
    if c not in alphabet:
        alphabet += c

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

def predict_sentence(model, start_text, alphabet, n):
    one_hot = text_to_one_hot(start_text, alphabet)
    for i in range(n):
        out = model.predict(np.array([one_hot[-n_input_chars:]]))
        one_hot = np.concatenate([one_hot, out], axis=0)
    return one_hot_to_text(one_hot, alphabet)

model = ks.models.load_model('./model-lstm.banaan')

while True:
    input_text = input("%d long text: "%n_input_chars)
    if len(input_text) != n_input_chars:
        print("learn to read idiot")
    else:
        print(predict_sentence(model, input_text, alphabet, 1000))


