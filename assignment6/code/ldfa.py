import sys
import numpy as np
from numpy.linalg import eig
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class LDFA(object):

    def __init__(self, X, y, classes):
        self.X = X
        self.y = y
        self.classes = classes
        self.W = np.empty((X.shape[1], 0))

    def train(self):
        A = self.__computeAffinity()
        S_W, S_B = self.__computeVariances(A)
        self.__updateW(S_W, S_B) # Generalized eigenvector problem

    # Use cosine similarity
    def __computeAffinity(self):
        N = self.X.shape[0]
        A_sparse = sparse.csr_matrix(self.X)
        A = cosine_similarity(A_sparse)
        return A

    def __computeVariances(self, A):
        N, d = self.X.shape
        N_k = {c: self.y[self.y==c].shape[0] for c in self.classes}
        S_W = np.zeros((d,d))
        S_B = np.zeros((d,d))
        for i in range(N):
            for j in range(N):
                diffX_i = (self.X[i] - self.X[j]).reshape((d,1))
                diffX_i_squared = np.matmul(diffX_i, diffX_i.T)
                if self.y[i] == self.y[j]:
                    S_W += (A[i,j]/N_k[self.y[i]])*diffX_i_squared
                    S_B += A[i,j]*((1/N)-(1/N_k[self.y[i]]))*diffX_i_squared
                else:
                    S_B += (1/N)*diffX_i_squared
        return S_W, S_B

    def __updateW(self, S_W, S_B):
        eig_val, eig_vec = eig(np.matmul(np.linalg.inv(S_W), S_B))

        # check eig_val
        eig_val_dict = {}
        for i in range(len(eig_val)):
            eig_val_dict[i] = eig_val[i]
        eig_val_sorted = sorted(eig_val_dict.items(), key=itemgetter(1), reverse=True)

        # Stack together
        k = len(self.classes)-1
        for i in range(k):
            toStack = eig_vec[:, eig_val_sorted[i][0]].reshape((self.W.shape[0],1))
            self.W = np.hstack((self.W, toStack))
    

