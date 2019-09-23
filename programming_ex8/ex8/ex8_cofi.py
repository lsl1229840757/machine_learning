import scipy.io as scio
from matplotlib import pyplot as plt
import cofiCostFunc
import numpy as np
import checkNNGradients
import scipy.optimize as opt

def serialize(X, Theta):
    return np.concatenate((X.flatten(), Theta.flatten()))


def deserialize(params, num_users, num_movies, num_features):
    X = np.reshape(params[0:num_movies * num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies * num_features:], (num_users, num_features))
    return X, Theta


# 协同过滤
# ######################## loading movie ratings dataset ################
if __name__ == "__main__":
    # load moving ratings dataset
    data = scio.loadmat("../machine-learning-ex8/ex8/ex8_movies.mat")
    Y = data["Y"]
    R = data["R"]
    print(Y)
    # visualize Y data
    plt.imshow(Y)
    # plt.show()
    movie_params = scio.loadmat("../machine-learning-ex8/ex8/ex8_movieParams.mat")
    # reduce datset
    num_users = 4
    num_movies = 5
    num_features = 3
    X = movie_params["X"][0:num_movies, 0:num_features]
    R = R[0:num_movies, 0:num_users]
    Y = Y[0:num_movies, 0:num_users]
    Theta = movie_params["Theta"][0:num_users, 0:num_features]
    params = serialize(X, Theta)
    cost, grad = cofiCostFunc.cotiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
    print("cost is {}, expected cost is 31.34".format(cost))
    grad_numerically = checkNNGradients.compute_grad_numerically(params, Y, R, num_users, num_movies, num_features, 1.5)
    print("grad is {}, grad_numerically is {}".format(grad, grad_numerically))
    # ========= parse movie_ids =======================
    movie_list = []
    with open("../machine-learning-ex8/ex8/movie_ids.txt", encoding='latin-1') as file:
        lines = file.readlines()
        for line in lines:
            token = line.strip().split(" ")
            movie_list.append(' '.join(token[1:]))
        movie_list = np.array(movie_list)
    # reproduce my ratings
    ratings = np.zeros(1682)
    ratings[0] = 4
    ratings[6] = 3
    ratings[11] = 5
    ratings[53] = 4
    ratings[63] = 5
    ratings[65] = 3
    ratings[68] = 5
    ratings[97] = 2
    ratings[182] = 4
    ratings[225] = 5
    ratings[354] = 5
    # get Y, R
    Y = data["Y"]
    R = data["R"]
    # now I become user0
    Y = np.insert(Y, 0, ratings, axis=1)
    R = np.insert(R, 0, ratings != 0, axis=1)  # type casting*****
    # some params
    n_features = 50
    n_movie, n_user = Y.shape
    l = 10
    X = np.random.standard_normal((n_movie, n_features))
    Theta = np.random.standard_normal((n_user, n_features))
    param = serialize(X, Theta)
    # normalized ratings
    Y_mean = np.reshape(np.mean(Y, 1), (Y.shape[0], 1))
    normalized_Y = Y - Y_mean

    res = opt.minimize(fun=cofiCostFunc.regularized_cost,
                       x0=param,
                       args=(normalized_Y, R, n_user, n_movie, n_features, l),
                       method='TNC',
                       jac=cofiCostFunc.regularized_gradient)
    print(res)
    X_train, Theta_train = deserialize(res.x, n_user, n_movie, n_features)
    prediction = X_train@Theta_train.T
    my_preds = prediction[:, 0].reshape((-1, 1)) + Y_mean  # I'm user 0
    idxs = np.argsort(my_preds.flatten())[::-1][:10]  # Descending order
    for m in movie_list[idxs]:
        # show top 10
        print(m)
