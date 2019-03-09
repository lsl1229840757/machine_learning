import numpy as np
from scipy.special import expit
# compute the gradient of the sigmoid function evaluated at z


def sigmoid_gradient(z):
    return np.multiply((1-expit(z)), expit(z))
