import numpy as np
import costFunction


# x_data contains x0
def gradient_descent(theta, x_data, y_data):
    iters = 200000
    for i in range(iters):
        grad = costFunction.gradient(theta, x_data, y_data)  # grad is a vector
        grad.reshape((-1, 1))
        alpha = 1e-3
        theta -= alpha*grad
        if (i+1) % 1000 == 0:
            print("with {} iterations,cost is {}".format(i+1, costFunction.cost_function(theta, x_data, y_data)))
    return theta.flatten()
