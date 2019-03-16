import numpy as np


def get_cost(theta, x_data, y_data, l):
    theta_tempt = theta.copy()
    theta_tempt = theta_tempt.reshape((-1, 1))
    m = x_data.shape[0]
    x_data_tempt = np.insert(x_data, 0, 1, axis=1)  # add x0
    hx = x_data_tempt @ theta_tempt
    part1 = np.sum(np.power(hx-y_data, 2)) / (m * 2)
    theta_tempt[0] = 0
    regularization = l / (2 * m) * (theta_tempt.T @ theta_tempt)
    return part1+regularization


def get_grad(theta, x_data, y_data, l):
    m = x_data.shape[0]
    theta_tempt = theta.copy()
    theta_tempt = theta_tempt.reshape((-1, 1))
    x_data_tempt = np.insert(x_data, 0, 1, axis=1)  # add x0
    hx = x_data_tempt @ theta_tempt
    hy = hx - y_data
    theta_tempt[0] = 0
    grad = (x_data_tempt.T @ hy + l * theta_tempt) / m
    return grad.flatten()

