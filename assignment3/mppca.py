import sys
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal as mvn
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
    m = 3
    W = np.random.rand(d,q,m)                      # Need to be properly initialized
    Var = np.random.rand(1,m)+1000000000000                    # Need to be properly initialized
    pi = np.random.rand(1,m)
    pi = pi / pi.sum()
    mu = np.random.rand(d,m)

    # Run EM algorithm until converges
    # Need to plot to see what's going on
    for i in range(5):
        R = np.empty(shape=(N,0))
        R_denominator = 0
        for j in range(m):
            print(Var)
            y = mvn(mean=mu[:,j] , cov=Var[0][j])
            R_denominator += pi[0][j]*y.pdf(np.transpose(T))
        
        R_denominator = R_denominator.reshape((N,1))

        for j in range(m):
            y = mvn(mean=mu[:,j] , cov=Var[0][j])
            nominator = pi[0][j]*y.pdf(np.transpose(T)).reshape((N,1))
            R = np.append(R, nominator/R_denominator, axis=1)

        pi, mu, W, Var = EM_MPPCA(W, Var[0], pi[0], mu, R, d, q, N, T, m)

    print('pi: ', pi)
    print('mu: ', mu)
    print('W: ', W)
    print('Var: ', Var)

    # 1d Ploting
    # var = 1
    # x_ = 0
    # for i in range(m):
    #     x_ += pi[0][i]*np.matmul(np.transpose(W[:,:,i]), T)
    # x_1 = x_[0][0:50]
    # plt.plot(np.zeros_like(x_1) + var, x_1, '.', color='r')
    # x_2 = x_[0][50:100]
    # plt.plot(np.zeros_like(x_2) + var, x_2, '.', color='b')
    # x_3 = x_[0][100:150]
    # plt.plot(np.zeros_like(x_3) + var, x_3, '.', color='g')
    # plt.show()

    # 2d Ploting
    x_ = 0
    for i in range(m):
        x_ += pi[0][i]*np.matmul(np.transpose(W[:,:,i]), T)
    plt.plot(x_[0][0:59], x_[1][0:59], '.', color='r')
    plt.plot(x_[0][59:130], x_[1][59:130], '.', color='b')
    plt.plot(x_[0][130:178], x_[1][130:178], '.', color='g')
    plt.show()


"""
Description: One step of EM algorithm for MPPCA
Input: S, W_old, Var_old
Output: W_new, Var_new
"""
def EM_MPPCA(W_old, Var_old, pi_old, mu_old, R, d, q, N, T, m):
    pi_new = R.sum(axis=0) / N
    mu_new = np.matmul(T, R) / R.sum(axis=0)

    S = np.empty(shape=(d,d,0))
    for j in range(m):
        S_t = 0
        for n in range(N):
            S_t += R[n][j]*np.matmul(T[:,[n]]-mu_new[:,[j]], np.transpose(T[:,[n]]-mu_new[:,[j]]))
        S_t = S_t / (pi_new[j]*N)
        S = np.dstack((S, S_t))

    M = np.empty(shape=(q,q,0))
    for j in range(m):
        W_old_t = W_old[:,:,j]
        M_t = np.matmul(np.transpose(W_old_t), W_old_t) + np.identity(q)*Var_old[j]
        M = np.dstack((M, M_t))

    W_new = np.empty(shape=(d,q,0))
    for j in range(m):
        W_old_t = W_old[:,:,j]
        W_new_temp = np.identity(q)*Var_old[j] + \
                     np.matmul(np.matmul(np.matmul(inv(M[:,:,j]), \
                     np.transpose(W_old_t)), S[:,:,j]), W_old_t)
        W_new_t = np.matmul(np.matmul(S[:,:,j], W_old_t), inv(W_new_temp))
        W_new = np.dstack((W_new, W_new_t))

    Var_new = np.empty(shape=(1,0))
    for j in range(m):
        W_old_t = W_old[:,:,j]
        W_new_t = W_new[:,:,j]
        Var_new_temp = np.matmul(np.matmul(np.matmul(S[:,:,j], W_old_t), inv(M[:,:,j])), \
                                 np.transpose(W_new_t))
        Var_new_t = (1/d) * np.trace(S[:,:,j] - Var_new_temp)
        Var_new = np.append(Var_new, Var_new_t)
    Var_new = Var_new.reshape((1,m))
    
    return pi_new.reshape((1,m)), mu_new, W_new, Var_new

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

