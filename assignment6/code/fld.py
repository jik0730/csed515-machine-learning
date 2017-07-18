import sys
import numpy as np
from numpy.linalg import eig
from operator import itemgetter

class FLD(object):

    def __init__(self, X, y, classes):
        self.X = X
        self.y = y
        self.classes = classes
        self.W = np.empty((X.shape[1], 0))

    def train(self):
        S_W, S_B = self.S()
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

    # Outputs dx1
    def __mean(self, X):
        return np.mean(X, axis=0)

    # Outputs S_W(dxd), S_B(dxd)
    def S(self):
        S_W = 0
        S_B = 0
        m = self.__mean(self.X)
        for c in self.classes:
            X_c = self.X[self.y==c]
            m_c = self.__mean(X_c)
            S_i = np.matmul((X_c-m_c).T, (X_c-m_c))
            ss = (m_c-m).reshape((self.X.shape[1],1))
            S_i2 = X_c.shape[0] * np.matmul(ss, ss.T)
            S_W += S_i
            S_B += S_i2
        return S_W, S_B

