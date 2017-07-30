import os
import keras as ks
import numpy as np

from util.text import filter_text, text_to_one_hot

word_length = 20

def split_pad_text(text):
    text = text.replace("\n", " ")
    text = text.split(" ")
    for i in range(len(text)):
        if len(text[i]) > word_length:
            text[i] = text[i][:word_length]
        elif len(text[i]) < word_length:
            text[i] += " " * (word_length - len(text[i]))

    return text

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

#alphabet = ['e', 'y', 'f', 's', 'o', 'r', 'n', 'p', '0', 'j', 'k', '1', 'd', '8', 'i', 'c', '6', '2', 'b', 'w', 'z', ' ', '5', 'v', 't', 'u', 'x', 'h', '4', 'q', 'g', 'l', '3', '7', 'a', '9', 'm']
alphabet = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

with open('../../../sources/other/temp.txt', 'r') as f:
    words = filter_text(f.read())
    words = split_pad_text(words)

x = np.array([text_to_one_hot(word, alphabet) for word in words])
pred = model.predict(x)
print(pred)
print("Avg:", np.mean(pred, axis=0))