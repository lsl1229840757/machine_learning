import numpy as np
from scipy.optimize import minimize
import linearRegCostFunction


def train(x_data, y_data, l):
    theta = np.zeros((x_data.shape[1]+1, ))
    result = minimize(linearRegCostFunction.get_cost, theta, method="TNC",
                      jac=linearRegCostFunction.get_grad, args=(x_data, y_data, l))
    return result.x
