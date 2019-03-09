import numpy as np


def initialize(fan_out, fan_in):
    w = np.zeros((fan_out, fan_in+1))
    tempt = np.sin(np.array(range(1, w.size+1, 1)))
    w = np.reshape(tempt, w.shape) / 10
    return w
