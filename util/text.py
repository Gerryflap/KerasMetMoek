import re
import numpy as np


def filter_text(text, pattern="[^a-z0-9 ]"):
    """
    Filters text.
    Test is put in lowercase and everything matching pattern will be removed from the string.
    :param text: Input text
    :param pattern: Optional pattern that selects the substrings that should be deleted
    :return: Filtered text
    """
    text = text.lower()
    text = re.sub(pattern, "", text)
    return text

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