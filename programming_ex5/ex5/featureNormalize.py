import numpy as np


def feature_scaling(x_data):
    mu = np.mean(x_data, axis=0)
    std = np.std(x_data, axis=0)
    result = (x_data - mu) / std
    return result, mu, std


# ignore x0
def feature_normalize(x_data):
    x_data = np.insert(x_data, 0, 1, axis=1)
    mu = np.mean(x_data, 0)
    std = np.std(x_data, 0)
    x_data[:, 1:] -= mu[1:]
    x_data[:, 1:] /= std[1:]
    return x_data[:, 1:], mu[1:], std[1:]
