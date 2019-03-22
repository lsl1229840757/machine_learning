import numpy as np


def guassian_kernel(x1, x2, sigma=2):
    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))
    return np.exp((-(x1-x2).T @ (x1-x2))/(2 * sigma**2))
