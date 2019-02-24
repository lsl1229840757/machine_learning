import matplotlib.pyplot as plt
import numpy as np
import mapFeature


def plot_decesion_boundary(theta, x_data, y_data):
    # plotData focus : x is 100 * 3 matrix
    pos = np.where(y_data == 1)
    neg = np.where(y_data == 0)
    plt.plot(x_data[pos[0], 1], x_data[pos[0], 2], 'k+')
    plt.plot(x_data[neg[0], 1], x_data[neg[0], 2], 'ko', color="y")
    plt.xlabel("exam1 score")
    plt.ylabel("exam2 score")
    plt.legend(["Admitted", "Not admitted"], loc='upper right')

    # ignore the number of colum = 2
    # when the number of colum = 3
    if x_data.shape[1] == 3:
        # two points can define a line
        plot_x = np.zeros((2, ))
        plot_x[0] = np.min(x_data[:, 1])
        plot_x[1] = np.max(x_data[:, 2])
        # calculate plot_y
        plot_y = -(theta[1]*plot_x+theta[0])/theta[2]
        plt.plot(plot_x, plot_y)
        plt.show()
    elif x_data.shape[1] > 3:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.ones((u.shape[0], v.shape[0]))
        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                tempt = np.column_stack((np.ones((1, 1)), mapFeature.map_feature(np.array([u[i]]), np.array([v[j]]))))
                a = np.dot(tempt, theta)
                z[i, j] = a
        plt.contour(u, v, z, [0], colors='k')
        plt.show()
