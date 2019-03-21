import numpy as np
import matplotlib.pyplot as plt
import plotData
from scipy.io import loadmat
# ========= Part1:loading data and visualizing data ==========================
data = loadmat("../machine-learning-ex6/ex6/ex6data1.mat")
# for i in data.keys():
#     print(i)
x_data = data["X"]
y_data = data["y"]
plotData.plot(x_data, y_data)
# ========= Part2:train linear SVM ===========================================
