import matplotlib.pyplot as plt


def plot(x_data, y_data):
    print('Plotting Data ...')
    plt.xlabel("Population of city in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.title("Scatter plot of training data")
    plt.scatter(x_data, y_data, marker="*", edgecolors="red")
    plt.show()
    return plt
