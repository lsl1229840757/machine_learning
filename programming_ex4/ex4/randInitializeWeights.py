import numpy as np


def inintialize_weights(l_in, l_out):
    # initialize some params
    epsilon = 0.12
    weights = np.random.rand(l_out, l_in+1) * 2 * epsilon - epsilon  # map the weights into [-epsilon,epsilon], add bias
    return weights


