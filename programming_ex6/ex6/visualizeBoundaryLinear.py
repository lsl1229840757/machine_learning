import numpy as np


def visualize_boundary(my_svm, xmin, xmax, ymin, ymax, plt):
    xvals = np.linspace(xmin, xmax, 100)
    yvals = np.linspace(ymin, ymax, 100)
    zvals = np.zeros((len(yvals), len(xvals)))
    # focus zvals is different from it in matlab
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[j][i] = my_svm.predict(np.array([[xvals[i], yvals[j]]]))
    plt.contour(xvals, yvals, zvals, [0])
    plt.show()

