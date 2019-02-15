import numpy as np
import computeCost


def gradient_descent(x_data, y_data, theta, alpha, iterations):
    # compute the derivation of theta and return cost_history
    # initiate some useful params
    cost_history = np.zeros((iterations, 1))
    for i in range(iterations):
        delta_theta = np.dot(x_data.T, np.dot(x_data, theta) - y_data)/y_data.shape[0]
        theta -= alpha * delta_theta
        cost_history[i, 0] = computeCost.compute_cost(x_data, y_data, theta)
        if (i+1) % 100 == 0:
            print("{}th iteration,cost:{}".format(i+1,cost_history[i, 0]))
    return theta, cost_history
