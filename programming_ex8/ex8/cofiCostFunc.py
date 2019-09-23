import numpy as np
from ex8_cofi import serialize, deserialize


# 计算误差J和梯度grad
def cotiCostFunc(params, Y, R, num_users, num_movies, num_features, l=0):
    # unfold params
    X, Theta = deserialize(params, num_users, num_movies, num_features)
    cost = predict(X, Theta, Y, R, l)
    # compute the partial derivatives to X
    return cost, regularized_gradient(params, Y, R, num_users, num_movies, num_features, l)


def regularized_gradient(params, Y, R, num_users, num_movies, num_features, l=0):
    X, Theta = deserialize(params, num_users, num_movies, num_features)
    X_grad = ((X @ Theta.T - Y) * R) @ Theta + l * X
    Theta_grad = ((X @ Theta.T - Y) * R).T @ X + l * Theta
    return serialize(X_grad, Theta_grad)


def regularized_cost(params, Y, R, num_users, num_movies, num_features, l=0):
    X, Theta = deserialize(params, num_users, num_movies, num_features)
    return predict(X, Theta, Y, R, l)


def predict(X, Theta, Y, R, l=0):
    reg_term = (np.sum(np.power(X, 2)) + np.sum(np.power(Theta, 2)))/2*l
    return np.sum(np.power((X@Theta.T - Y)*R, 2))/2 + reg_term
