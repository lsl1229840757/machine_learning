import numpy as np



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_reg(theta, x_data, y_data, l):
    m, n = x_data.shape
    theta = theta.reshape((-1, 1))
    y_data = y_data.reshape((-1, 1))
    hx = sigmoid(np.dot(x_data, theta))
    ln_h = np.log(hx)
    part1 = -np.dot(ln_h.T, y_data) / m
    ln_1h = np.log(1 - hx)
    part2 = -np.dot(ln_1h.T, 1-y_data) / m
    #   don't do that : theta[0, 0] = 0  # don't penalize theta0
    reg = l * np.dot(theta[1:, :].T, theta[1:, :]) / (2 * m)
    return (part1+part2+reg).flatten()


def grad_reg(theta, x_data, y_data, l):
    m, n = x_data.shape
    theta_tempt = theta.reshape((-1, 1)).copy()
    y_data = y_data.reshape((-1, 1))
    hx = sigmoid(np.dot(x_data, theta_tempt))
    theta_tempt[0, 0] = 0  # don't penalize theta0
    part1 = np.dot(x_data.T, hx - y_data)
    part2 = l * theta_tempt
    return ((part1+part2) / m).flatten()

