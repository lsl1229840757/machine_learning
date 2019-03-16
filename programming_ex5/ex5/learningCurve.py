import numpy as np
import linearRegCostFunction
import trainLinearReg


def get_errors(x_train, y_train, x_val, y_val, l=0):
    m, n = x_train.shape
    errors_train = np.zeros((m, 1))
    errors_val = np.zeros((m, 1))
    for i in range(m):
        theta = trainLinearReg.train(x_train[0:i+1, :], y_train[0:i+1, :], l)
        # get train errors
        errors_train[i, :] = linearRegCostFunction.get_cost(theta, x_train[0:i+1, :], y_train[0:i+1, :], l)
        # get cross validation errors
        errors_val[i, :] = linearRegCostFunction.get_cost(theta, x_val, y_val, l)  # need to go through all cv data to get errors
    return errors_train, errors_val
