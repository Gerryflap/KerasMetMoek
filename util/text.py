import re
import numpy as np


def filter_text(text, pattern=None, allow_numbers=False, extended_latin=True, allow_new_line=False):
    """
    Filters text.
    Test examples put in lowercase and everything matching pattern will be removed from the string.
    :param text: Input text
    :param pattern: Optional pattern that selects the substrings that should be deleted
    :return: Filtered text
    """
    # text = text.replace("-", " ")
    if (pattern is None):
        pattern = create_pattern(allow_numbers=allow_numbers, extended_latin=extended_latin, allow_new_line=allow_new_line)
    else:
        pattern = pattern
    text = text.replace("\n", " ")
    text = text.lower()
    text = re.sub(pattern, "", text)
    return text

def create_pattern(allow_numbers=False, extended_latin=True, allow_new_line=False):
    pattern = "[^a-z "
    if (allow_numbers):
        pattern += "0-9"
    if (extended_latin):
        pattern += "\u00C0-\u024F"
    if (allow_new_line):
        pattern += "\n"
    pattern += "]"
    return pattern


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