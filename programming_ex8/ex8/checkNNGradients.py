import numpy as np
import ex8_cofi
import cofiCostFunc


def compute_grad_numerically(params, Y, R, num_users, num_movies, num_features, l=0):
    # compute gradient numerically
    # prepare result ndarray
    epsilon = 1e-4
    grad = np.zeros(params.shape)
    # deserialize params
    for i in range(params.size):
        paramsr_tempt = params.copy()
        paramsr_tempt[i] += epsilon
        paramsl_tempt = params.copy()
        paramsl_tempt[i] -= epsilon
        Xr, Thetar = ex8_cofi.deserialize(paramsr_tempt, num_users, num_movies, num_features)
        Xl, Thetal = ex8_cofi.deserialize(paramsl_tempt, num_users, num_movies, num_features)
        costr = cofiCostFunc.predict(Xr, Thetar, Y, R, l)
        costl = cofiCostFunc.predict(Xl, Thetal, Y, R, l)
        grad[i] = (costr-costl)/(2*epsilon)
    return grad
