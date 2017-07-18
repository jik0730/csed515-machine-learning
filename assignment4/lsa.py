import sys
from scipy.io import loadmat
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from nltk.cluster.kmeans import KMeansClusterer
import nltk
from itertools import permutations


"""
Description: Implement Latent Semantic Analysis
"""
def main(matfile):
    # Import data
    T, X = dataPreprocessing(matfile)

    # SVD
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    # eigenvalues_plotting = plt.figure()
    # plt.plot(range(475), S)
    # plt.title('eigenvalue SNOWFALL matrix')
    # plt.ylabel('eigenvalue')

    energy = 0
    energy_plot = []
    accuracy_plot = []
    for k in range(10):
        # SVD
        U, S, VT = np.linalg.svd(X, full_matrices=False)

        # Low rank approximation
        S_sum = np.sum(S)
        VT_index = len(S) - 1
        energy += 0.1               # Eigenvalue energy
        energy_plot.append(energy)
        S_percent = 0
        for i in range(len(S)):
            S_percent += S[i]
            if S_percent / S_sum >= energy:
                VT_index = i
                break
        VT = VT[0:VT_index,:]

        """ Other k-means algorithm 
            # K-means
            # kmeans = KMeans(n_clusters=4, max_iter=1000000000).fit(np.transpose(VT))
            # result = kmeans.labels_

            # K-means by cv2
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000000, 1.0)
            # ret,result,center = cv2.kmeans(np.float32(np.transpose(VT)),2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        """

        file = open("lsa_"+str(energy)+"_.txt", "w")
        c = 0
        for i in range(100):
            # K-means by nltk
            NUM_CLUSTERS = 4
            data = np.transpose(np.matmul(np.diag(S[0:VT_index]),VT))
            kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
            result = kclusterer.cluster(data, assign_clusters=True)

            # Analysis
            # f1 = plt.figure()
            # plt.plot(range(101), result[0:101], '.', color='r')
            # f2 = plt.figure()
            # plt.plot(range(101,172), result[101:172], '.', color='g')
            # f3 = plt.figure()
            # plt.plot(range(172,350), result[172:350], '.', color='b')
            # f4 = plt.figure()
            # plt.plot(range(350,475), result[350:475], '.', color='y')
            # plt.plot(range(475), result, '.')
            plt.show()

            a = accuracy(result)
            c += a
            file.write(str(i+1)+","+str(a)+","+str(c/(i+1))+"\n")
            # print('accuracy is %s' %(a))
        
        print('Energy %s mean accuracy is %s' %(energy, c/(i+1)))
        accuracy_plot.append(c)
        file.close()

    plt.plot(energy_plot, accuracy_plot)
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
Description: Retrieve data from matfile
Input: matfile
Output: nclass, ndoc, nterm, T, X
"""
def dataPreprocessing(matfile):
    matDict = loadmat(matfile)
    T = matDict['T']
    X = matDict['term_doc']
    return T, X


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Failed to start a program')
    else:
        main(sys.argv[1])

