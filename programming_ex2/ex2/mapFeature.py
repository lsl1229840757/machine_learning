import numpy as np


# return [ x1,x1*x2 ... ] the 27-dimension vector, ignore 1
def map_feature(x1, x2):
    m = x1.shape[0]
    result = np.zeros((m, 27))
    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))
    end = 0
    for i in range(1, 7):
        for j in range(0, i+1):
            result[:, end] = np.multiply(np.power(x1[:, 0], i-j), np.power(x2[:, 0], j))
            end += 1
    return result