import numpy as np


# compute the sigmoid of data, data can be a vector, matrix or scalar
def sigmoid(data):
    return 1/(1+np.exp(-data))
