import numpy as np
import lrCostFunction

def pred_lr(theta, x_data):
    x_data = x_data.reshape((1, -1))
    x_data = np.column_stack((np.ones((1, 1)), x_data))
    result = lrCostFunction.sigmoid(x_data @ theta.T)
    predict = np.argmax(result.flatten())
    if predict == 0:
        predict = 10
    return predict


def pred_nn(theta1, theta2, x_data):
    m, _ = x_data.shape
    a1 = x_data
    a1 = np.column_stack((np.ones((m, )), a1))
    z2 = a1 @ theta1.T
    a2 = lrCostFunction.sigmoid(z2)
    a2 = np.column_stack((np.ones((a2.shape[0], )), a2))
    z3 = a2 @ theta2.T
    a3 = lrCostFunction.sigmoid(z3)
    a3 = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention
    return a3
