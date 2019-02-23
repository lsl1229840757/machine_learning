import numpy as np
import costFunction
import plotData
# ========================== Load Data =========================
m = 0
x_col_num = 0
with open("../machine-learning-ex2/ex2/ex2data1.txt") as file:
    line_list = file.readlines()
    m = len(line_list)
    feature_num = len(line_list[0].split(",")) - 1  # ignore x0 ,minus y
    # initialize x_data, y_data
    x_data = np.zeros((m, feature_num))
    y_data = np.zeros((m, 1))
    # assign value to x_data,y_data
    for i in range(m):
        line_tempt = line_list[i].split(",")
        x_data[i, :] = line_tempt[:feature_num]
        y_data[i, 0] = line_tempt[-1]
# ========================= Load end =============================
# ========================= plotData =============================
plotData.plot_data(x_data, y_data)
# ========================= compute cost and gradient ===========
# prepare x_data and initiate theta
n = x_data.shape[1]
x_data = np.column_stack((np.ones((m, 1)), x_data))
initial_theta = np.zeros((n+1, 1))
cost = costFunction.cost_function(initial_theta, x_data, y_data)
# print("initial_theta cost is {} (approx)".format(cost))
# print("expected cost is 0.693")
# print("initial_theta grad is {}".format(grad)) grad can be gotten from  costFunction.gradient
# print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
# ============= Part 3: Optimizing using fminunc(matlab)/scipy(python)  =============
# theta must be a vector
#theta = op.minimize(costFunction.cost_function, initial_theta, args=(x_data, y_data), method="TNC", jac=costFunction.gradient)
#print(theta)
