'''
主要是理解几个概念：
    1.假设函数
    2.代价函数
    3.梯度下降
'''
import numpy as np
import matplotlib.pyplot as plt
'''
 使用梯度下降的方法实现线性拟合,不使用向量的方式直接循环完成,返回一次函数的参数列表由theta0->theta1
'''
# 在这里我会使用resource下面的txt文件实现单变量线性拟合:通过梯度下降和正规方程两种方式进行拟合


def gradient_descent(x_data, y_data,learing_ratio):
    k = np.random.rand(1)
    b = np.random.rand(1)
    iterations = 0
    # 为了实现实时反应error
    iterations_list = []
    error_list = []
    # 进行1000次迭代
    while iterations < 5000:
        error = 0
        delta_k = 0
        delta_b = 0
        for i in range(len(x_data)):
            # 写tempt让过程更明显
            error_tempt = (line_formula(k, b, x_data[i]) - y_data[i])**2/2
            error += error_tempt
            # 完成对k,b的更新,手动推出关于k，b的偏导数（也可以利用偏导数定义直接计算）
            delta_k_tempt = (line_formula(k, b, x_data[i]) - y_data[i])*x_data[i]
            delta_b_tempt = line_formula(k, b, x_data[i]) - y_data[i]
            delta_k += delta_k_tempt
            delta_b += delta_b_tempt
        delta_k /= len(x_data)
        delta_b /= len(x_data)
        error /= len(x_data)
        k -= learing_ratio * delta_k
        b -= learing_ratio * delta_b
        # 每1000次画一个图,debug:从图中可以看出在迭代20000次之后便会收敛(学习率为1e-3时),4000次收敛（学习率为1e-2）
        if (iterations+1) % 1000 == 0:
            error_list.append(error)
            iterations_list.append(iterations+1)
            plt.clf()
            plt.ioff()
            plt.plot(iterations_list, error_list )
            plt.show()
        iterations += 1
    return [k, b]


# 使用正规方程组求解
def normal_equation(x_data, y_data):
    x_data = np.array(x_data)
    x_data = x_data[:, np.newaxis]
    x_data = np.column_stack((x_data, np.ones(len(x_data))))
    y_data = np.array(y_data)
    y_data = y_data[:,np.newaxis]
    return np.dot(np.dot(np.linalg.pinv(np.dot(x_data.T, x_data)), x_data.T), y_data)


def line_formula(k, b, x):
    return k*x+b


if __name__ == '__main__':
    # 读取数据
    file = open("./resource/ex1data1.txt", "r")
    x_data = []
    y_data = []
    for line in file.readlines():
        tempt_list = line.split(",")
        x_data.append(float(tempt_list[0]))
        y_data.append(float(tempt_list[1]))
    params = gradient_descent(x_data, y_data, 1e-2)
    print("梯度下降result:{}".format(params))
    params2 = normal_equation(x_data,y_data)
    print("正规方程result:{}".format(params2))
