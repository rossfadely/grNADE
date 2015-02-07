import numpy as np

def softmax(x):
    """
    1D softmax function
    """
    v = np.exp(x - x.max())
    return v / np.sum(v)
