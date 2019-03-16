import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import linearRegCostFunction
import trainLinearReg
import learningCurve
import polyFeatures
import featureNormalize
# ================= loading and visualize dataset ===================================
dataset = loadmat("../machine-learning-ex5/ex5/ex5data1.mat")
# get all keys
# for i in dataset.keys():
#     print(i)
# get all training data
x_train = dataset["X"]
y_train = dataset["y"]
x_val = dataset["Xval"]
y_val = dataset["yval"]
x_test = dataset["Xtest"]
y_test = dataset["ytest"]
m = x_train.shape[0]
plt.scatter(x_train, y_train)
plt.xlabel("Change in water level(x)")
plt.ylabel("Water flowing out of dam")
# ================ Regularized Linear regression cost ===============
theta = np.array([1, 1])
J = linearRegCostFunction.get_cost(theta, x_train, y_train, 1)
print("with theta = [1;1],l=1 cost is {}".format(J))
print("expected cost is 303.993192")
# =============== Regularized linear regression grad ================
grad = linearRegCostFunction.get_grad(theta, x_train, y_train, 1)
print("with theta = [1;1],l=1 grad is {}".format(grad))
print("expected grad is [-15.303016; 598.250744]")
# =============== train linear regression params ========================
# minimize cost with lambda = 0
l = 0
final_theta = trainLinearReg.train(x_train, y_train, l)
# plot fit data
x_fit = np.insert(x_train, 0, 1, axis=1)
y_fit = x_fit @ final_theta
plt.plot(x_train, y_fit)
plt.show()
# ================ learning curve for linear regression ===================
errors_train, errors_val = learningCurve.get_errors(x_train, y_train, x_val, y_val, 0)
plt.figure()
plt.plot(np.arange(1, m+1), errors_val)
plt.plot(np.arange(1, m+1), errors_train)
plt.legend(("errors_val", "errors_train"))
plt.xlabel("the size of trainDataSet")
plt.ylabel("errors")
plt.show()
# ================ feature mapping for polynomial regression ===============
###############################################################
# My d=8 plot doesn't match the homework pdf, due to differences
# between scipy.optimize.fmin_cg and the octave version
# I see that in subokita's implementation, for fitting he gets the
# same results as I when using scipy.optimize.fmin_cg
#
# The d=5 plot (up through x^6) shows overfitting clearly, so I'll
# continue using that
###############################################################
p = 5
# map train data
x_poly_train = polyFeatures.map_feature(x_train, p)
x_poly_train, mu, std = featureNormalize.feature_scaling(x_poly_train)  # normalize the data
# map test data using mu and std
x_poly_test = polyFeatures.map_feature(x_test, p)
x_poly_test = (x_poly_train - mu) / std
# map cross validation data
x_poly_val = polyFeatures.map_feature(x_val, p)
x_poly_val = (x_poly_val - mu) / std
result_poly_theta = trainLinearReg.train(x_poly_train, y_train, l)
# plot fitting curve
plt.figure()
plt.scatter(x_train, y_train)
x_plot = np.linspace(-55, 55, 50).reshape((-1, 1))
x_plot_nomal = polyFeatures.map_feature(x_plot, p)
x_plot_nomal = (x_plot_nomal - mu) / std
plt.plot(x_plot, np.insert(x_plot_nomal, 0, 1, axis=1)@result_poly_theta.reshape((-1, 1)))
plt.xlabel("Change in water level")
plt.ylabel("water flowing out of dam")
plt.show()
# ======================  learning curve for polynomial regression =======================
# test different l
l = 1
plt.figure()
train_errors, cv_errors = learningCurve.get_errors(x_poly_train, y_train, x_poly_val, y_val, l)
plt.plot(np.arange(1, m+1), cv_errors)
plt.plot(np.arange(1, m+1), train_errors)
plt.legend(("errors_val", "errors_train"))
plt.show()
# ====================== select the best lambda ============================================
mylambda = np.linspace(0, 5, 20)
train_errors_list = np.zeros((20, ))
cv_errors_list = np.zeros((20, ))
count = 0
for l in mylambda:
    theta = trainLinearReg.train(x_poly_train, y_train, l)
    train_errors = linearRegCostFunction.get_cost(theta, x_poly_train, y_train, 0)  # to get cost, l = 0
    cv_errors = linearRegCostFunction.get_cost(theta, x_poly_val, y_val, 0)
    train_errors_list[count] = train_errors
    cv_errors_list[count] = cv_errors
    count += 1
plt.plot(mylambda, cv_errors_list)
plt.plot(mylambda, train_errors_list)
plt.legend(("errors_val", "errors_train"))
plt.xlabel("lambda")
plt.ylabel("errors")
plt.show()
