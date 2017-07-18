import sys
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


"""
Description: Main function
"""
def main(arg1):
    # Get data set
    T = getDataSet(arg1)

    # Set parameters
    d = np.shape(T)[0]
    N = np.shape(T)[1]
    q = 2                                          # Need to be changed
    W = np.random.rand(d,q)                      # Need to be properly initialized
    Var = np.random.rand()                    # Need to be properly initialized
    # mu = np.random.rand(d,1)
    mu = np.mean(T, axis=1).reshape((d,1))
    S = 0
    for i in range(N):
        S += np.matmul(T[:,[i]]-mu, np.transpose(T[:,[i]]-mu))
    S /= N

    # Run EM algorithm until converges
    # Need to plot to see what's going on
    for i in range(100):
        W_old = W
        Var_old = Var
        W, Var = EM_PPCA(S, W, Var, d, q)

    # 1d Ploting
    # var = 1
    # x_ = np.matmul(np.transpose(W), T)
    # x_1 = x_[0][0:59]
    # print (x_1)
    # plt.plot(np.zeros_like(x_1) + var, x_1, '.', color='r')
    # x_2 = x_[0][59:130]
    # plt.plot(np.zeros_like(x_2) + var, x_2, '.', color='b')
    # x_3 = x_[0][130:178]
    # plt.plot(np.zeros_like(x_3) + var, x_3, '.', color='g')
    # plt.show()

    # 2d Ploting
    x_ = np.matmul(np.transpose(W), T)
    print (x_)
    plt.plot(x_[0][0:59], x_[1][0:59], '.', color='r')
    plt.plot(x_[0][59:130], x_[1][59:130], '.', color='b')
    plt.plot(x_[0][130:178], x_[1][130:178], '.', color='g')
    plt.show()



"""
Description: One step of EM algorithm for PPCA
Input: S, W_old, Var_old
Output: W_new, Var_new
"""
def EM_PPCA(S, W_old, Var_old, d, q):
    W_old_T = np.transpose(W_old)
    M = np.matmul(W_old_T, W_old) + np.identity(q)*Var_old
    W_new_temp = np.identity(q)*Var_old + \
                 np.matmul(np.matmul(np.matmul(inv(M), W_old_T), S), W_old)
    W_new = np.matmul(np.matmul(S, W_old), inv(W_new_temp))
    Var_new_temp = np.matmul(np.matmul(np.matmul(S, W_old), inv(M)), \
                             np.transpose(W_new))
    Var_new = (1/d) * np.trace(S - Var_new_temp)
    
    return W_new, Var_new

"""
Description: Open a dataset file
Input: filename
Output: T(dataset)
"""
def getDataSet(filename):
    T = np.empty(shape=(13,0))
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datum = line.split(',')
            datum = list(map(float, datum[1:14]))
            T = np.append(T, np.array(datum).reshape((13,1)), axis=1)
    return T

if __name__ == "__main__":
    main(sys.argv[1])

