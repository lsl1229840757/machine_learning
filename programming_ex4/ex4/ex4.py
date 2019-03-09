# ==================== exercise 4 Neural Network learning ============================
import numpy as np
from scipy.io import loadmat
import scipy.optimize as op
import displayData
import nnCostFunction
import sigmoidGradient
import randInitializeWeights
import checkNNGradients
# initialize some params
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
# ======================= Loading data and visualizing =================================
data = loadmat("../machine-learning-ex4/ex4/ex4data1.mat")
X = data["X"]
y_data = data["y"]
x_data = [im.reshape((20, 20)).T for im in X]
x_data = np.array([im.reshape((400, )) for im in x_data])  # don't reshape im to be 2-dimension matrix
randperm = np.random.randint(0, 5000, (100, ))
display_data = x_data[randperm, :]
displayData.display(display_data)
# ===================== load end ========================================================
nn_data = loadmat("../machine-learning-ex4/ex4/ex4weights.mat")
theta1 = nn_data["Theta1"]
theta2 = nn_data["Theta2"]
# ===================== test cost =======================================================
# unroll params
nn_params = np.column_stack((theta1[:].reshape((1, -1)), theta2[:].reshape((1, -1)))).flatten()
J = nnCostFunction.nn_cost(nn_params, input_layer_size, hidden_layer_size, X, y_data, num_labels)
print("cost is {}".format(J))
print("expected cost is 0.287629")
l = 1
J_reg = nnCostFunction.nn_cost_reg(nn_params, input_layer_size, hidden_layer_size, X, y_data, num_labels, l)
print("cost_reg is {}".format(J_reg))
print("expected cost_reg is 0.383770")
# =================== test end ==========================================================
# =================== sigmoid gradient =================================================
g = sigmoidGradient.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print("sigmoid gradient is {}".format(g))
# =================== sigmoid gradient end ====================================================
# =================== Initialize params =======================================================
initial_theta1 = randInitializeWeights.inintialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights.inintialize_weights(hidden_layer_size, num_labels)
# unroll params
initial_nn_params = np.column_stack((initial_theta1[:].reshape((1, -1)), initial_theta2[:].reshape((1, -1)))).flatten()
# =================== implements BackPropagation =============================================
# check grad
# checkNNGradients.check()
# =================== optimize the loss ======================================================
# result = op.minimize(nnCostFunction.nn_cost_reg, initial_nn_params, args=(input_layer_size, hidden_layer_size, x_data, y_data, num_labels, l), method="TNC", jac=nnCostFunction.nn_grad_reg)
# result = op.minimize(nnCostFunction.nn_cost_reg, initial_nn_params, args=(input_layer_size, hidden_layer_size, x_data, y_data, num_labels, l), method="TNC", jac=checkNNGradients.grad_num)
# print(result)