# ================== Multi-class Classification | part 1: one vs all ========================
from scipy.io import loadmat  # read matfile
import numpy as np
import displayData
import lrCostFunction
import oneVsAll
import Predict
import matplotlib.pyplot as plt
import predictOneVsAll
# ================== load Data =============================================================
data = loadmat("../machine-learning-ex3/ex3/ex3data1.mat")
x_data = data["X"]   # x_data here is a 5000 * 4000 matrix (5000 training examples, 20 * 20 grid of pixel is unrolled into 4000-deminsional vector
y_data = data.get("y")  # y_data here is a 5000 * 1 matrix(label)
# hint : must transpose the data to get the oriented data
x_data = np.array([im.reshape((20, 20)).T for im in x_data])
x_data = np.array([im.reshape((400, )) for im in x_data])
print(y_data)
# can use data.key() and debugging to get this information
# ================== load end ==============================================================
# set some params
input_layer_size = 400
num_labels = 10
# ================== visualize the data ====================================================
rand = np.random.randint(0, 5000, (100, ))  # [0, 5000)
displayData.data_display(x_data[rand, :])   # get 100 images randomly
# ======================= Test case for lrCostFunction =============================
theta_t = np.array([-2, -1, 1, 2])
t = np.linspace(1, 15, 15) / 10
t = t.reshape((3, 5))
x_t = np.column_stack((np.ones((5, 1)), t.T))
y_t = np.array([1, 0, 1, 0, 1])
l_t = 3
cost = lrCostFunction.cost_reg(theta_t, x_t, y_t, l_t)
grad = lrCostFunction.grad_reg(theta_t, x_t, y_t, l_t)
print("cost is {}".format(cost))
print("expected cost is 2.534819")
print("grad is {}".format(grad))
print("expected grad is 0.146561 -0.548558 0.724722 1.398003")
# ============================ test end =============================================
# ============================ one vs all:predict ===========================================
l = 0.1
theta = oneVsAll.one_vs_all(x_data, y_data, l, num_labels)
result = Predict.pred_lr(theta, x_data[1500, :])
np.set_printoptions(precision=2, suppress=True)  # don't use  scientific notation
print("this number is {}".format(result))  # 10 here is 0
plt.imshow(x_data[1500, :].reshape((20, 20)), cmap='gray', vmin=-1, vmax=1)
plt.show()
accuracy = predictOneVsAll.pred_accuracy(theta, x_data, y_data)
print("test 5000 images, accuracy is {:%}".format(accuracy))
# ============================ predict  end ======================================================
