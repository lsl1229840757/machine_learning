from scipy.io import loadmat  # read matfile
import numpy as np
import displayData
import Predict
# =========== Part 1: Loading and Visualizing Data =============
data = loadmat("../machine-learning-ex3/ex3/ex3data1.mat")
x_data = data["X"]   # to meet the weights params, we don't transpose x_data
y_data = data.get("y")  # y_data here is a 5000 * 1 matrix(label)
input_layer_size = 400
hidden_layer_size = 25   # 25 hidden units
num_labels = 10
# we must transpose x_data to show the image correctly
x = np.array([im.reshape((20, 20)).T for im in x_data])
x = np.array([im.reshape((400, )) for im in x])
rand = np.random.randint(0, 5000, (100, ))  # [0, 5000)
displayData.data_display(x[rand, :])   # get 100 images randomly
# ================== Loading end ================================
# ================ Part 2: Loading Parameters ================
weights = loadmat("../machine-learning-ex3/ex3/ex3weights.mat")
theta1 = weights["Theta1"]
theta2 = weights["Theta2"]
# ================ loading end ===============================
# ================= predict ==================================
result = Predict.pred_nn(theta1, theta2, x_data)
# compute accuracy
right_num = 0
for i in range(result.shape[0]):
    if result[i] == y_data[i, :]:
        right_num += 1
accuracy = right_num / result.shape[0]
print("test 5000 images accuracy is {:%}".format(accuracy))
