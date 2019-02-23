import matplotlib.pyplot as plt
import numpy as np


def plot_data(x_data, y_data):
    pos = np.where(y_data == 1)
    neg = np.where(y_data == 0)
    plt.plot(x_data[pos[0], 0], x_data[pos[0], 1], 'k+')
    plt.plot(x_data[neg[0], 0], x_data[neg[0], 1], 'ko', color="y")
    plt.xlabel("exam1 score")
    plt.ylabel("exam2 score")
    plt.legend(["Admitted", "Not admitted"], loc='upper right')
    plt.show()
