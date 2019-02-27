import numpy as np
import matplotlib.pyplot as plt


def data_display(data, image_width=20):
    # compute image_height
    m, n = data.shape
    image_height = n / image_width
    # compute rows and cols
    display_rows = np.round(np.sqrt(m))
    display_cols = m / display_rows
    # set image padding
    pad = 1
    rows_pixel = np.ceil(display_rows * image_height + (display_rows + 1) * pad)
    cols_pixel = np.ceil(display_cols * image_width + (display_cols + 1) * pad)
    rows_pixel = rows_pixel.astype(np.int64)
    cols_pixel = cols_pixel.astype(np.int64)
    # initialize display matrix
    display_matrix = -np.ones((rows_pixel, cols_pixel))
    # the first pixel of every image is 1+(image_width+pad)*(n-1) or 1+(image_height+pad)*(n-1)
    for i in range(data.shape[0]):
        image_data = data[i, :].reshape((int(image_width), int(image_height)))
        row_index = np.floor(i / display_cols)
        cols_index = i % display_cols
        row_position = pad+(image_height+pad)*row_index
        cols_position = pad+(image_width+pad)*cols_index
        display_matrix[int(row_position):int(row_position+image_height), int(cols_position):int(cols_position+image_width)] = image_data[:, :]
    plt.imshow(display_matrix, cmap='gray', vmin=-1, vmax=1)
    plt.show()


