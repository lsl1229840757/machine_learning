import numpy as np
import plotData_reg
import mapFeature
import costFunctionReg
import scipy.optimize as op
import plotDecisionBoundary
file = open("../machine-learning-ex2/ex2/ex2data2.txt")
# initialize some params
line_list = file.readlines()
m = len(line_list)
n = len(line_list[0].split(","))-1  # ignore x0
x_data = np.zeros((m, n))
y_data = np.zeros((m, 1))
for i in range(m):
    line_tempt = line_list[i].split(",")
    x_data[i, :] = line_tempt[:2]
    y_data[i, :] = line_tempt[-1]
file.close()
# ================== plotData ===================
plotData_reg.plot_data(x_data,  y_data)
# ================== plotData end ===============
# ================== regularized logistic regression ==============
# ================== map feature ================================
map_x = mapFeature.map_feature(x_data[:, 0], x_data[:, 1])
x_data = np.column_stack((np.ones((m, 1)), map_x))
# ================== costFunctionReg ==========================
# set regulation params
lambd = 10  # you can try to change the lambda(0, 1, 10 ,100 or others),when the lambda is 1,the result is best
theta = np.ones((x_data.shape[1]))
# ====== test cost and grad =========  lambda = 10
# cost = costFunctionReg.cost_function(theta, x_data, y_data, lambd)
# print("cost is {}".format(cost))
# print("expected cost is 3.16")
grad = costFunctionReg.gradient(theta, x_data, y_data, lambd)
print("grad is {}".format(grad[0:5]))
print("0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n")
# ====== test end ====================================
result = op.minimize(costFunctionReg.cost_function, theta, args=(x_data, y_data, lambd), method="TNC", jac=costFunctionReg.gradient)
print(result)
final_theta = result.x
# ========== plot boundary ===========
plotDecisionBoundary.plot_decesion_boundary(final_theta, x_data, y_data)
