import matplotlib.pyplot as plt
import numpy as np


def display(data, image_rows_num=10):
    m, n = data.shape
    pad = 1
    image_rows_pixel = int(np.ceil(np.sqrt(n)))
    image_cols_pixel = int(np.ceil(n / image_rows_pixel))
    image_cols_num = int(np.ceil(m / image_rows_num))  # compute the num of image in colums
    pixel_rows = image_rows_num * image_rows_pixel + (image_rows_num + 1) * pad
    pixel_cols = image_cols_num * image_cols_pixel + (image_cols_num + 1) * pad
    display_matrix = - np.ones((pixel_rows, pixel_cols))
    for i in range(m):
        im = data[i, :].reshape((image_rows_pixel, -1))
        cols = i % image_cols_num
        rows = int(i / image_cols_num)
        cols_index = pad + cols * (pad + image_cols_pixel)
        rows_index = pad + rows * (pad + image_rows_pixel)
        display_matrix[int(rows_index):int(rows_index+image_rows_pixel), int(cols_index):int(cols_index+image_cols_pixel)] = im
    plt.imshow(display_matrix, cmap='gray', vmin=-1, vmax=1)
    plt.show()
