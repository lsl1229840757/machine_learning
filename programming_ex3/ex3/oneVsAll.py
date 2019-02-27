import numpy as np
import fmincg

def one_vs_all(x_data, y_data, l, num_labels):
    # initialize some params
    m, n = x_data.shape
    x_data = np.column_stack((np.ones((m, 1)), x_data))
    # initialize initial_theta
    initial_thata = np.zeros((num_labels, n+1))
    theta = fmincg.fmincg(initial_thata, x_data, y_data, l, num_labels)
    return theta
