import keras as ks
import numpy as np
import sys

from util.text import text_to_one_hot, filter_text

word_length = 20

def split_pad_text(text):
    text = text.split(" ")
    for i in range(len(text)):
        if len(text[i]) > word_length:
            text[i] = text[i][:word_length]
        elif len(text[i]) < word_length:
            text[i] += " " * (word_length - len(text[i]))
    # Remove words that consist only of spaces
    text = [word for word in text if word[0] != " "]
    return text

alphabet = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ß', 'ä', 'ö', 'ü']
model = ks.models.load_model("./best_model_goed.hdf5")
print("READY", flush=True)
while True:
    line = sys.stdin.readline()
    sys.stderr.write("Received line!\n")
    sys.stderr.flush()
    line = filter_text(line)
    x = np.array([text_to_one_hot(word, alphabet) for word in split_pad_text(line)])
    output = model.predict(x)
    avg_output = np.average(output, axis=0)
    print("PRED\t%f\t%f"%(avg_output[0], avg_output[1]), flush=True)
