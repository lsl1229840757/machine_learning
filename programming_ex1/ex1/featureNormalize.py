import numpy as np


# ignore x0
def feature_normalize(x_data):
    mu = np.mean(x_data, 0)
    std = np.std(x_data, 0)
    x_data[:, 1:] -= mu[1:]
    x_data[:, 1:] /= std[1:]
    return x_data, mu, std
