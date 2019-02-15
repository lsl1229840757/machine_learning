import numpy as np


# compute with matrix not with iterations
def compute_cost(x_data, y_data, theta):
    return np.sum(np.power(np.dot(x_data, theta)-y_data, 2)/2)/y_data.shape[0]
