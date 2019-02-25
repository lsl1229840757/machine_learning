import numpy as np
import sigmoid


# focus:ignore theta0 in regularization
def cost_function(theta, x_data, y_data, lambd):
    theta = theta.reshape((-1, 1))
    m = x_data.shape[0]
    h_x = sigmoid.sigmoid(np.dot(x_data, theta))
    ln_x = np.log(h_x)
    ln_x1 = np.log(1-h_x)
    part1 = np.dot(ln_x.T, y_data)/m
    part2 = np.dot(ln_x1.T, (1-y_data))/m
    regularization = lambd*np.dot(theta[1:, :].T, theta[1:, :])/(2*m)
    return -part1-part2+regularization


def gradient(theta, x_data, y_data, lambd):
    theta = theta.reshape((-1, 1))
    m = x_data.shape[0]
    h_x = sigmoid.sigmoid(np.dot(x_data, theta))
    theta[0, 0] = 0  # don't penalize theta0
    part1 = np.dot(x_data.T, h_x-y_data)/m
    part2 = lambd/m*theta
    return (part1+part2).flatten()
