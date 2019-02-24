import matplotlib.pyplot as plt
import numpy as np


def plot_data(x_data, y_data):
    pos = np.where(y_data == 1)
    neg = np.where(y_data == 0)
    plot_x = x_data[pos[0], 0]
    plot_y = x_data[pos[0], 1]
    plot_x1 = x_data[neg[0], 0]
    plot_y1 = x_data[neg[0], 1]
    plt.plot(plot_x, plot_y, 'k+')
    plt.plot(plot_x1, plot_y1, 'ko', color="y")
    plt.xlabel("Microchip test1")
    plt.ylabel("Microchip test2")
    plt.legend(("y=1", "y=0"))
    plt.show()
