import numpy as np
import sigmoid


# compute the cost and the partial derivations of theta, x,y can be a vector, matrix, scalar
def cost_function(theta, x_data, y_data):
    m = y_data.shape[0]
    theta = theta.reshape((-1, 1))
    # compute cost
    h_x = sigmoid.sigmoid(np.dot(x_data, theta))
    ln_h = np.log(h_x)  # np.log() is the Natural Log
    part1 = -np.dot(ln_h.T, y_data) / m
    ln_1h = np.log(1-h_x)
    part2 = -np.dot(ln_1h.T, 1-y_data) / m
    cost = part1+part2
    # compute the partial derivations of theta
    return cost


# because scipy.op.minimize is different from fminunc, we need a method returns the gradient independently
def gradient(theta, x_data, y_data):
    theta = theta.reshape((-1, 1))
    m = y_data.shape[0]
    # compute cost
    h_x = sigmoid.sigmoid(np.dot(x_data, theta))
    # compute the partial derivations of theta
    grad = np.dot(x_data.T, h_x - y_data) / m
    return grad.flatten()  # must be a vector
