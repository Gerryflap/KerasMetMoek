import keras as ks
import numpy as np
import matplotlib.pyplot as plt

n_input_chars = 5
#text = "abababababababababababababab"
text_original = "Hallo, je wist het waarschijnlijk al: ik ben een banaan. Je zult je wellicht afvragen, wat doet een banaan zoal? Bananen zijn voornamelijk bezig met liggen of hangen. Er zijn echter situaties waar een banaan opgegeten wordt of opgegeten is. Dat is wat een banaan doet."
text_original = text_original.lower()
text_fake = "banaan is een banaan"
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


one_hot_text = text_to_one_hot(text_original, alphabet)
print(one_hot_text)

model = ks.models.load_model('./model-lstm.banaan')

t_dim = np.reshape(one_hot_text, [-1, 1, len(alphabet)])
# Generate 5 long sequences
x = np.concatenate([t_dim[i:len(t_dim)-(n_input_chars)+i] for i in range(n_input_chars)], axis=1)
y = np.reshape(t_dim[n_input_chars:], [-1, len(alphabet)])

# try:
#     history = model.fit(x, y, epochs=1000, batch_size=256)
# except KeyboardInterrupt:
#     print("Training interrupted")

predicted_text = np.reshape(model.predict(x), [-1, len(alphabet)])

print(one_hot_to_text(predicted_text, alphabet))
print(one_hot_to_text(y, alphabet))
# model.save("./model-lstm.banaan")

print(alphabet_frequency(one_hot_to_text(predicted_text, alphabet), alphabet))
print(alphabet_frequency(one_hot_to_text(y, alphabet), alphabet))

# plt.plot(np.log(history.history['loss']))
# plt.show()

