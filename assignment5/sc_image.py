import sys
import scipy.io
import numpy as np
import scipy as sp
from scipy.linalg import eigh
from scipy.linalg import eig
import math
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import euclidean_distance
import matplotlib.pyplot as plt
from PIL import Image
import time


"""
Description: Main function implementing spectral clustering with image
segmentation
"""
def main(image, affinity=None, diagonal=None, eigen=None):
    # Read flattened image data(n*d)
    data, n, m, d = readImage(image)
    k = 3

    # Unnormalized Laplacian(n*n)
    start_time = time.time()
    L, D, W = unnormalizedLaplacian(data, affinity, diagonal)
    print("Compute Unnormalized Laplacian \
        --- %s seconds ---" % (time.time() - start_time))

    # Generalized eigenvectors(n*k)
    start_time = time.time()
    # U = generalizedEigenvectors(L, D, k, eigen)
    U = generalizedEigenvectorsByNA(W, 10, k)
    print("Compute Generalized eigenvectors \
        --- %s seconds ---" % (time.time() - start_time))

    # K-means clustering in U
    start_time = time.time()
    result, means = kmeans(U, k)
    print("K-means clustering in U \
        --- %s seconds ---" % (time.time() - start_time))

    # Show result plot
    showResult(n, m, d, k, result, data)


"""
Description: Show results
"""
def showResult(n, m, d, k, result, data):
    img = np.empty((0, d), dtype=np.uint8)
    means = np.zeros((k,3))
    counts = np.zeros((k,1))
    for i in range(n*m):
        means[result[i]] += data[i]
        counts[result[i]] += 1
    means = means/counts
    for i in range(n*m):
        img = np.append(img, means[result[i]])
    img = img.reshape((n, m, d))
    # print(img)
    print(means)
    print(counts)
    imgResult = Image.fromarray(np.uint8(img))
    imgResult.show()


"""
Description: K-means for spectral clustering
"""
def kmeans(data, k):
    # K-means by nltk
    NUM_CLUSTERS = k
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=euclidean_distance, repeats=25, avoid_empty_clusters=True)
    result = kclusterer.cluster(data, assign_clusters=True)
    return result, kclusterer.means()


"""
Description: Compute unnormalized Laplacian
Output: Graph Laplacian
WAY TOO SLOWWWW!!!WAY TOO SLOWWWW!!!WAY TOO SLOWWWW!!!WAY TOO SLOWWWW!!!
"""
def unnormalizedLaplacian(data, affinity, diagonal):
    if affinity == None or diagonal == None:
        n = data.shape[0]
        W = np.zeros([n,n])
        D = np.zeros([n,n])
        for i in range(n):
            d = 0
            for j in range(i):
                d += W[i,j]
            for j in range(i,n):
                x1, y1, z1 = data[i]
                x2, y2, z2 = data[j]
                dist = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
                if dist == 0:
                    continue
                else:
                    W[i,j] = np.exp(-dist/5000)
                    W[j,i] = W[i,j]
                d += W[i,j]
            D[i,i] = d
            print(i)
        np.save('./affinity.npy', W)
        np.save('./diagonal.npy', D)
        # W = np.subtract.outer(data, data)
        # print(W.shape)
        # W = np.linalg.norm(W, ord=2, axis=2)
        # print(W.shape)
        # W = np.exp(-W)
        # print(W.shape)
        # sys.exit()
    else:
        W = np.load(affinity)
        D = np.load(diagonal)

    return D-W, D, W


"""
Description: Compute generalized eigenvalue problem
Output: U with k eigenvectors
"""
def generalizedEigenvectors(L, D, k, eigen):
    if eigen == None:
        w, v = eigh(L, D, eigvals=(0,k-1))
        np.save('./eigen.npy', v)
    else:
        v = np.load(eigen)
    return v


"""
Description: Compute generalized eigenvalue problem by Nystrom approx.
Output: U with k eigenvectors
"""
def generalizedEigenvectorsByNA(W, n, k):
    N = W.shape[0]
    A = W[0:n, 0:n] + np.random.rand(n,n)/100000
    B = W[0:n, n:N]
    BT = np.transpose(B)
    print(A)
    A_ = np.linalg.inv(np.sqrt(A))

    Q = A + np.matmul(np.matmul(np.matmul(A_, B), BT), A_)
    print(Q)
    w, U = eigh(Q)
    print(w)
    w_ = np.linalg.inv(np.diag(w))

    V = np.matmul(np.matmul(np.matmul(W[:, 0:n], A_), U), w_)
    print(V)

    return V
    


"""
Description: Read image
Input: image name
Output: numpy.ndarray(n*3)
"""
def readImage(image):
    img = Image.open(image)
    imgArray = PIL2array(img)
    n, m, d = imgArray.shape
    return imgArray.reshape((imgArray.shape[0]*imgArray.shape[1],
                             imgArray.shape[2])), n, m, d

def PIL2array(img):
    return np.array(img.getdata(),
                    np.int64).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print('Wrong arguments')
        

