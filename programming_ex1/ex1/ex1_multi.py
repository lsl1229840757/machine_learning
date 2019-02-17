import numpy as np
import featureNormalize
import gradientDescent
# ================ Part 1: Feature Normalization ================
# loading data
file = open("../machine-learning-ex1/ex1/ex1data2.txt", "r")
line_list = file.readlines()
m = len(line_list)
# compute the size of x_data, y_data
x_col_num = len(line_list[0].split(","))  # minus y_data plus x0
# initialize x_data and y_data
x_data = np.zeros((m, x_col_num))
y_data = np.zeros((m, 1))
x_data[:, 0] = 1
for i in range(m):
    line_tempt = line_list[i].split(",")
    for j in range(len(line_tempt)):
        if j != len(line_tempt)-1:
            x_data[i, j+1] = line_tempt[j]
        else:
            y_data[i, 0] = line_tempt[j]
# print(x_data)
# print(y_data)
file.close()
# description:x_data is a m*3 matrix, y_data is a vector
# loading end
# ignore the feature x0
x_data, mu, std = featureNormalize.feature_normalize(x_data)
#  trick!!: normalize data before adding x0
# =======================Part2. Gradient Descent ===========================
# initialize some params
alpha = 0.01
num_iters = 5000
theta = np.zeros((3, 1))
# my gradientDescent support multi-variables
theta, cost_history = gradientDescent.gradient_descent(x_data, y_data, theta, alpha, num_iters)
# print(theta)
# =======================Part3. Predict ================
input_data = np.array([
    [1, 1650, 3]
], dtype=np.float64)
input_data[0, 1:] -= mu[1:]
input_data[0, 1:] /= std[1:]
print("predict price is {}".format(np.dot(input_data, theta)))
# 正规方程比较简单，我在1_linear_regression_with_one_variable中写过，这里就不再写了
# 有几点要注意的：
# 1. 梯度下降使用了特征缩放，所以theta应该和正规方程不一样。
# 2. 预测的时候不要忘了特征缩放。
