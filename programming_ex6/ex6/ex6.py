import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import plotData
from scipy.io import loadmat
import visualizeBoundaryLinear
import guassianKernel
# ========= Part1:loading data and visualizing data ==========================
data = loadmat("../machine-learning-ex6/ex6/ex6data1.mat")
# for i in data.keys():
#     print(i)
x_data = data["X"]
y_data = data["y"]
plotData.plot(x_data, y_data, plt)
# ========= Part2:train linear SVM ===========================================
# use sklearn, different from the lecture
linear_svm = svm.SVC(C=1, kernel="linear")
linear_svm.fit(x_data, y_data.flatten())
plt.title("use linear kernel to fit linear regression, c=1")
visualizeBoundaryLinear.visualize_boundary(linear_svm, 0, 4, 1.5, 5, plt)
# try c = 100
plotData.plot(x_data, y_data, plt)
linear_svm = svm.SVC(C=100, kernel="linear")
linear_svm.fit(x_data, y_data.flatten())
plt.title("use linear kernel to fit linear regression, c=100")
visualizeBoundaryLinear.visualize_boundary(linear_svm, 0, 4, 1.5, 5, plt)
# ============== Part 3: Implementing Gaussian Kernel ===============
# test guassianKernel
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
test_result = guassianKernel.guassian_kernel(x1, x2, sigma)
print("test result is {}".format(test_result))
print("expected value is 0.324652")
plotData.plot(x_data, y_data, plt)
guassian_svm = svm.SVC(C=1, kernel=guassianKernel.guassian_kernel)  # there must be tow input params
guassian_svm.fit(x_data, y_data.flatten())
plt.title("use guassian kernel to fit linear regression")
visualizeBoundaryLinear.visualize_boundary(linear_svm, 0, 4, 1.5, 5, plt)
# =============== Part 4: Visualizing Dataset 2 ================
data2 = loadmat("../machine-learning-ex6/ex6/ex6data2.mat")
x_data2 = data2["X"]
y_data2 = data2["y"]
plotData.plot(x_data2, y_data2, plt)
# use custom kernel
# guassian_svm = svm.SVC(C=1, kernel=guassianKernel.guassian_kernel)
# guassian_svm.fit(x_data2, y_data2.flatten())
# plt.title("use use custom guassian kernel")
# visualizeBoundaryLinear.visualize_boundary(guassian_svm, 0, 1, 0.4, 1, plt)
# plt.show()
sigma = 0.1
gamma = np.power(sigma, -2.)
gaus_svm = svm.SVC(C=1, kernel='rbf', gamma=gamma)
gaus_svm.fit(x_data2, y_data2.flatten())
visualizeBoundaryLinear.visualize_boundary(gaus_svm, 0, 1, .4, 1.0, plt)
# =============== Part 6: Visualizing Dataset 3 ================
data3 = loadmat("../machine-learning-ex6/ex6/ex6data3.mat")
x_data3, y_data3 = data3["X"], data3["y"]
xval, yval = data3["Xval"], data3["yval"]
values = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
best_svm = 0
best_score = 0
best_c = 0
best_sigma = 0
for c in values:
    for sigma in values:
        gamma = np.power(sigma, -2)
        g_svm = svm.SVC(c, "rbf", gamma=gamma)
        g_svm.fit(x_data3, y_data3.flatten())
        score = g_svm.score(xval, yval.flatten())
        if score > best_score:
            best_svm = g_svm
            best_c = c
            best_sigma = sigma
            best_score = score
plotData.plot(x_data3, y_data3, plt)
plt.title("best scores:{} with c={},sigma={}".format(best_score, best_c, best_sigma))
visualizeBoundaryLinear.visualize_boundary(best_svm, -.5, .3, -.8, .6, plt)
