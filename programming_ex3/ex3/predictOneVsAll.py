import numpy as np
import Predict

def pred_accuracy(theta, x_data, y_data):
    right_num = 0
    m, _ = x_data.shape
    for i in range(m):
        pred = Predict.pred_lr(theta, x_data[i, :])
        if pred == y_data[i, :]:
            right_num += 1
    return right_num / m
