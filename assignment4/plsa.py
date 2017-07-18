import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from itertools import permutations


"""
Description: main function
"""
def main(matfile):
    # Import data
    T, X = dataPreprocessing(matfile)
    
    # file = open("plsa_10_100_revised.txt", "w")
    # c = 0
    # for j in range(1):
    # Set parameters
    NUM_LATENT = 4
    t, d = X.shape
    Z, W_Z, D_Z = initParams(NUM_LATENT, t, d)

    # EM iteration
    for i in range(100):

        if (i+1) % 10 == 0:
            print('Iteration %s' %(i))
            result_temp = doc_Clustering(D_Z, Z, d, NUM_LATENT)
            a = accuracy(result_temp)
            print('accuracy is %s' %(a))

        EM_PLSA(X, Z, W_Z, D_Z, t, d, NUM_LATENT)

    # Document clustering
    result = doc_Clustering(D_Z, Z, d, NUM_LATENT)

    # Accuracy estimate
    # a = accuracy(result)
    # c += a
    # file.write(str(j+1)+","+str(a)+","+str(c/(j+1))+"\n")
    # print('accuracy is %s' %(a))
    # print('mean accuracy is %s' %(c/(j+1)))

    # file.close()

    # Plot
    plt.plot(range(101), result[0:101], '.', color='r')
    plt.plot(range(101,172), result[101:172], '.', color='g')
    plt.plot(range(172,350), result[172:350], '.', color='b')
    plt.plot(range(350,475), result[350:475], '.', color='y')
    plt.show()


"""
Description: Compute accuracy
"""
def accuracy(labels):
    a = labels[0:101]
    b = labels[101:172]
    c = labels[172:350]
    d = labels[350:475]
    max_accuracy = 0
    possible_permutations = permutations(range(4))
    for per in possible_permutations:
        count1 = a.count(per[0])
        count2 = b.count(per[1])
        count3 = c.count(per[2])
        count4 = d.count(per[3])
        accuracy = (count1+count2+count3+count4)/475
        if accuracy > max_accuracy:
            max_accuracy = accuracy
    return max_accuracy


"""
Description: Document clustering and return labels
"""
def doc_Clustering(D_Z, Z, d, NUM_LATENT):
    labels = []
    for j in range(d):
        maxValue = -1
        maxIndex = -1
        for k in range(NUM_LATENT):
            if D_Z[j][k]*Z[k] > maxValue:
                maxValue = D_Z[j][k]*Z[k]
                maxIndex = k
        labels.append(maxIndex)
    return labels


"""
Description: EM algorithm for PLSA / Update params
"""
def EM_PLSA(X, Z, W_Z, D_Z, t, d, NUM_LATENT):
    # E-step
    Z_WD = np.float128(np.empty(shape=(t,d,0)))
    WD_sum = 0
    WD_temp = []
    for k in range(NUM_LATENT):
        W_temp = W_Z[:,k].reshape((t,1))
        D_temp = np.transpose(D_Z[:,k].reshape((d,1)))
        WD = np.matmul(W_temp, D_temp)*Z[k]
        WD_sum += WD
        WD_temp.append(WD)
    
    for k in range(NUM_LATENT):
        WD = WD_temp[k] / WD_sum
        Z_WD = np.dstack((Z_WD, WD))
    Z_WD = Z_WD / np.sum(Z_WD, axis=2).reshape((t,d,1))

    # M-step
    # W_Z
    for k in range(NUM_LATENT):
        denumerator = 0
        for i in range(t):
            numerator = 0
            for j in range(d):
                numerator += X[i][j]*Z_WD[i][j][k]
            W_Z[i][k] = numerator
            denumerator += numerator
        W_Z[:,k] /= denumerator
    W_Z = W_Z / np.sum(W_Z, axis=0)

    # D_Z
    for k in range(NUM_LATENT):
        denumerator = 0
        for j in range(d):
            numerator = 0
            for i in range(t):
                numerator += X[i][j]*Z_WD[i][j][k]
            D_Z[j][k] = numerator
            denumerator += numerator
        D_Z[:,k] /= denumerator
    D_Z = D_Z / np.sum(D_Z, axis=0)

    # Z
    for k in range(NUM_LATENT):
        numerator = 0
        denumerator = 0
        for i in range(t):
            for j in range(d):
                numerator += X[i][j]*Z_WD[i][j][k]
                denumerator += X[i][j]
        Z[k] = numerator
        Z[k] /= denumerator
    Z = Z / np.sum(Z)


"""
Description: Initialize parameters
Output: Z, W_Z D_Z
"""
def initParams(NUM_LATENT, t, d):
    Z = np.random.rand(NUM_LATENT, 1)
    Z = Z / np.sum(Z)
    W_Z = np.random.rand(t, NUM_LATENT)
    W_Z = W_Z / np.sum(W_Z, axis=0)
    D_Z = np.random.rand(d, NUM_LATENT)
    D_Z = D_Z / np.sum(D_Z, axis=0)
    return np.float128(Z), np.float128(W_Z), np.float128(D_Z)


"""
Description: Retrieve data from matfile
Input: matfile
Output: nclass, ndoc, nterm, T, X
"""
def dataPreprocessing(matfile):
    matDict = loadmat(matfile)
    T = matDict['T']
    X = matDict['term_doc']
    return np.float128(T), np.float128(X)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Failed to start a program')
    else:
        main(sys.argv[1])