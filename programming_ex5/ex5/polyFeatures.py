import numpy as np


def map_feature(x_data, p):
    m = x_data.shape[0]
    x_poly = np.zeros((m, p))
    for i in range(p):
        x_poly[:, i] = np.power(x_data[:, 0], i+1)
    return x_poly
