import numpy as np
import matplotlib.pyplot as plt


# ======== define function ==========
# hypothesis function,batch
def hypothesis(x_data, theta):
    return x_data @ theta


# cost function
def cost_compute(x_data, y_data, theta):
    y_pred = hypothesis(x_data, theta)
    error = y_pred - y_data
    error = np.power(error, 2)
    m, _ = y_data.shape
    error = np.sum(error)/(2 * m)
    return error


# gradient_descent function
def gradient_descent(x_data, y_data, theta, alpha, iterations):
    history_cost = []
    m, _ = x_data.shape
    # begin iterations
    for i in range(iterations):
        delta_theta = x_data.T @ (hypothesis(x_data, theta) - y_data) / m
        theta = theta - alpha * delta_theta
        history_cost.append(cost_compute(x_data, y_data, theta))
    return theta, history_cost


# ======== plotting data ===========
if __name__ == "__main__":
    file = open("../machine-learning-ex1/ex1/ex1data1.txt")
    x_data = []
    y_data = []
    for line in file.readlines():
        line_split = line.split(",")
        x_data.append(float(line_split[0]))
        y_data.append(float(line_split[1]))
    file.close()
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    plt.scatter(x_data, y_data)
    plt.xlabel("population")
    plt.ylabel("profit")
    plt.title("population-profit")
    # prepare data
    x_data = x_data.reshape((-1, 1))
    y_data = y_data.reshape((-1, 1))
    # initialize some parameters
    m, n = x_data.shape
    x_data = np.column_stack((np.ones((m, 1)), x_data))
    theta = np.random.random((n+1, 1))
    alpha = 0.001
    iterations = 10000
    theta, history_cost = gradient_descent(x_data, y_data, theta, alpha, iterations)
    print(theta)
    print(history_cost)
    # visulize the line
    y_pred = hypothesis(x_data, theta)
    plt.plot(x_data[:, 1], y_pred, 'r')
    plt.show()
