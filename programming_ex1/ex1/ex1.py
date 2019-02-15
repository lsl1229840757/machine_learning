import warmUpExercise
import plotData
import computeCost
import gradientDescent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# ==================== Part 1: Basic Function ====================
warmUpExercise.warm_up_exercise()
# ======================= Part 2: Plotting =======================
file = open("../machine-learning-ex1/ex1/ex1data1.txt", "r")
x_data = []
y_data = []
for line in file.readlines():
    line_split = line.split(",")
    x_data.append(float(line_split[0]))
    y_data.append(float(line_split[1]))
plotData.plot(x_data, y_data)
# =================== Part 3: Cost and Gradient descent ===================
# initialize some useful variables
m = len(x_data)
theta = np.zeros((2, 1))
x_data = np.array(x_data)
y_data = np.array(y_data)
x_data = x_data[:, np.newaxis]
y_data = y_data[:, np.newaxis]
x_data = np.column_stack((np.ones(m), x_data))
# some gradient_descent settings
iterations = 1500
alpha = 0.01
# compute cost
J = computeCost.compute_cost(x_data, y_data, theta)
print("with theta is {},cost is {}".format(theta, J))
print("expected cost is 32.07")
# begin gradient descent
theta, cost_history = gradientDescent.gradient_descent(x_data, y_data, theta, alpha, iterations)
print("after {} iterations,theta is {}".format(iterations, theta))
print("expected theta is [-3.6303,1.1664]")
# plot the linear fit
plt.plot(x_data[:, 1], np.dot(x_data, theta))
plt.scatter(x_data[:, 1], y_data, marker="*", edgecolors="red")
plt.show()
# ============= Part 4: Visualizing J(theta_0, theta_1) =============
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        # hint: focus on the relation between theta0 and theta1 while plot3D
        t = np.vstack((theta0_vals[i], theta1_vals[j]))
        J_vals[i, j] = computeCost.compute_cost(x_data, y_data, t)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title("Visualizing J(theta_0, theta_1):surface")
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T, cmap='rainbow')
plt.figure()
plt.title("Visualizing J(theta_0, theta_1):contour")
plt.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20), cmap='rainbow')
plt.scatter(theta[0], theta[1], marker='*')
plt.show()
