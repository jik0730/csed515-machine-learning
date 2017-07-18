import sys
import scipy.io
import numpy as np
from scipy.linalg import eigh
import math
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import euclidean_distance
import matplotlib.pyplot as plt
import time


"""
Description: Main function implementing spectral clustering
"""
def main(matfile):
    # Read two_moons data(n*d)
    data = readMatFile(matfile)
    k = 2

    # Unnormalized Laplacian(n*n)
    L, D = unnormalizedLaplacian(data)

    # Generalized eigenvectors(n*k)
    U = generalizedEigenvectors(L, D, k)

    # Plot U
    # plotU(U)

    # K-means clustering in U
    result = kmeans(U, k)

    # Show result plot
    plot(data, result)


"""
Description: Plot U
"""
def plotU(U):
    f1 = plt.figure()
    plt.plot(U[:,0], U[:,1], '.')
    plt.show()


"""
Description: Plot results
"""
def plot(data, result):
    f2 = plt.figure()
    for i in range(len(result)):
        if result[i] == 0:
            plt.plot(data[i][0], data[i][1], '.', color='r')
        else:
            plt.plot(data[i][0], data[i][1], '.', color='b')
    plt.show()


"""
Description: K-means for spectral clustering
"""
def kmeans(data, k):
    # K-means by nltk
    NUM_CLUSTERS = k
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=euclidean_distance, repeats=25)
    result = kclusterer.cluster(data, assign_clusters=True)
    return result


"""
Description: Compute unnormalized Laplacian
Output: Graph Laplacian
"""
def unnormalizedLaplacian(data):
    n = data.shape[0]
    W = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            x1, y1 = data[i]
            x2, y2 = data[j]
            dist = (x1-x2)**2 + (y1-y2)**2
            W[i][j] = math.exp(-dist/0.04)

    D = np.zeros([n,n])
    for i in range(n):
        d = 0
        for j in range(n):
            d += W[i][j]
        D[i][i] = d

    return D-W, D


"""
Description: Compute generalized eigenvalue problem
Output: U with k eigenvectors
"""
def generalizedEigenvectors(L, D, k):
    # start_time = time.time()
    w, v = eigh(L, D, eigvals=(0,k-1))
    # print("--- %s seconds ---" % (time.time() - start_time))
    # np.save('./a', v)
    # vv = np.load('./a.npy')
    return v


"""
Description: Read matfile
Input: matfile name
Output: numpy.ndarray(200,2)
"""
def readMatFile(matfile):
    mat = scipy.io.loadmat(matfile)
    return mat['x']


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong arguments')
    else:
        main(sys.argv[1])