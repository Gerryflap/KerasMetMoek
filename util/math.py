import numpy as np

def deriv(x):
    return x[1:] - x[:-1]

def unison_shuffled_copies(a, b):
    """Shuffles two equal length arrays at once"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]