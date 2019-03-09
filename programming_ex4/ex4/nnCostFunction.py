import numpy as np
from scipy.special import expit
import sigmoidGradient


def cost_function(hx, y):
    return - y * np.log(hx) - (1-y) * np.log(1-hx)


# don't use regularization
def nn_cost(nn_params, input_layer_size, hidden_layer_size, x_data, y_data, num_labels):
    m = x_data.shape[0]
    y_data = y_data.reshape((-1, 1))
    # forward propagation
    theta1 = nn_params[0: hidden_layer_size * (input_layer_size+1)].reshape((hidden_layer_size, -1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size+1):].reshape((num_labels, -1))
    # use for loop to finish forward propagation
    sum_cost = 0
    for i in range(m):
        a1 = x_data[i, :].reshape((-1, 1))
        label = y_data[i, :]
        a1 = np.vstack((np.ones((1, 1)), a1))  # add bias
        z2 = theta1 @ a1
        a2 = expit(z2)
        a2 = np.vstack((np.ones((1, 1)), a2))  # add bias
        z3 = theta2 @ a2
        a3 = expit(z3)
        # compute cost
        cost = 0
        for j in range(a3.shape[0]):
            if j+1 == label:
                flag = 1
            else:
                flag = 0
            cost += cost_function(a3[j, :], flag)
        sum_cost += cost
    sum_cost /= m
    return sum_cost


def nn_cost_reg(nn_params, input_layer_size, hidden_layer_size, x_data, y_data, num_labels, l):
    sum_cost = nn_cost(nn_params, input_layer_size, hidden_layer_size, x_data, y_data, num_labels)
    m = x_data.shape[0]
    theta1 = nn_params[0: hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, -1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, -1))
    regularization = (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))*l/(2*m)  # don't penalize bias
    return sum_cost + regularization


def nn_grad_reg(nn_params, input_layer_size, hidden_layer_size, x_data, y_data, num_labels, l):
    m = x_data.shape[0]
    y_data = y_data.reshape((-1, 1))
    # forward propagation
    theta1 = nn_params[0: hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, -1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, -1))
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    # set delta3,delta2 to store some useful values
    delta3 = np.zeros((num_labels, 1))
    delta2 = np.zeros((hidden_layer_size, 1))
    for i in range(m):
        a1 = x_data[i, :].reshape((-1, 1))
        label = y_data[i, :]
        a1 = np.vstack((np.ones((1, 1)), a1))  # add bias
        z2 = theta1 @ a1
        a2 = expit(z2)
        a2 = np.vstack((np.ones((1, 1)), a2))  # add bias
        z3 = theta2 @ a2
        a3 = expit(z3)
        # compute delta3
        for j in range(z3.shape[0]):
            if j+1 == label:
                flag = 1
            else:
                flag = 0
            delta3[j, 0] += grad_function(a3[j, :], flag) / m * sigmoidGradient.sigmoid_gradient(z3[j, :])
        # compute delta2
        for m in range(delta3.shape[0]):
            for j in range(z2.shape[0]):
                delta2[j, 0] += delta3[m, :] * theta2[m, j+1] * sigmoidGradient.sigmoid_gradient(z2[j, :])  #a2 have bias
        # compute theta2_grad
        for j in range(theta2_grad.shape[0]):
            for k in range(theta2_grad.shape[1]):
                theta2_grad[j, k] += delta3[j, 0] * a2[k, 0]
        # compute theta1_grad
        for j in range(theta1_grad.shape[0]):
            for k in range(theta1_grad.shape[1]):
                theta1_grad[j, k] += delta2[j, 0] * a1[k, 0]
    # add regularization
    theta1_grad[:, 1:] += theta1[:, 1:] / m * l
    theta2_grad[:, 1:] += theta2[:, 1:] / m * l
    grad_theta = np.column_stack((theta1_grad.reshape((1, -1)), theta2_grad.reshape((1, -1)))).flatten()
    return grad_theta


def grad_function(a, y):
    return - y/a + (1-y) / (1-a)


def deserialize(seq):
#     """into ndarray of (25, 401), (10, 26)"""
    return seq[:5 * 4].reshape(5, 4), seq[5 * 4:].reshape(3, 6)


def feed_forward(theta, X):
    """apply to architecture 400+1 * 25+1 *10
    X: 5000 * 401
    """

    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]
    a1 = X
    a1 = np.column_stack((np.ones((5, 1)), a1))  # add bias
    z2 = a1 @ t1.T  # 5000 * 25
    a2 = np.insert(expit(z2), 0, np.ones(m), axis=1)  # 5000*26

    z3 = a2 @ t2.T  # 5000 * 10
    h = expit(z3)  # 5000*10, this is h_theta(X)

    return a1, z2, a2, z3, h  # you need all those for backprop


def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))


def gradient(theta, X, y):
    # initialize
    y = y.reshape(-1, 1)
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]

    delta1 = np.zeros(t1.shape)  # (25, 401)
    delta2 = np.zeros(t2.shape)  # (10, 26)

    a1, z2, a2, z3, h = feed_forward(theta, X)

    for i in range(m):
        a1i = a1[i, :]  # (1, 401)
        z2i = z2[i, :]  # (1, 25)
        a2i = a2[i, :]  # (1, 26)

        hi = h[i, :]    # (1, 10)
        yi = y[i, :]    # (1, 10)

        d3i = hi - yi  # (1, 10)

        z2i = np.insert(z2i, 0, np.ones(1))  # make it (1, 26) to compute d2i
        d2i = np.multiply(t2.T @ d3i, sigmoidGradient.sigmoid_gradient(z2i))  # (1, 26)

        # careful with np vector transpose
        delta2 += np.matrix(d3i).T @ np.matrix(a2i)  # (1, 10).T @ (1, 26) -> (10, 26)
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i)  # (1, 25).T @ (1, 401) -> (25, 401)

    delta1 = delta1 / m
    delta2 = delta2 / m
    return serialize(delta1, delta2)


def regularized_gradient(theta, X, y, l=1):
    """don't regularize theta of bias terms"""
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))
    t1, t2 = deserialize(theta)

    t1[:, 0] = 0
    reg_term_d1 = (l / m) * t1
    delta1 = delta1 + reg_term_d1

    t2[:, 0] = 0
    reg_term_d2 = (l / m) * t2
    delta2 = delta2 + reg_term_d2

    return serialize(delta1, delta2)
