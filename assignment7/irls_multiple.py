import sys
import numpy as np
from sklearn.preprocessing import normalize

def main():
    # Iris data
    X_iris = np.genfromtxt('./data/iris.data', delimiter=',', usecols=(0,1,2,3), dtype=float)
    y_iris = np.genfromtxt('./data/iris.data', delimiter=',', usecols=(4), dtype=str)
    y_iris[y_iris=='Iris-setosa'] = 0
    y_iris[y_iris=='Iris-versicolor'] = 1
    y_iris[y_iris=='Iris-virginica'] = 2
    y_iris = y_iris.astype(int)
    y_iris = one_hot(y_iris, 3)
    X_iris = normalize(X_iris)


    # Wine data
    X_wine = np.genfromtxt('./data/wine.data', delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13), dtype=float)
    y_wine = np.genfromtxt('./data/wine.data', delimiter=',', usecols=(0), dtype=int)
    X_wine = normalize(X_wine)
    y_wine -= 1
    y_wine = one_hot(y_wine, 3)

    # IRLS for binary case
    w_iris = multipleIRLS(X_iris, y_iris)
    w_wine = multipleIRLS(X_wine, y_wine)

    # Evaluate accuracy of logistic regression
    accuracy_iris = multipleLogisticRegression(X_iris, y_iris, w_iris)
    accuracy_wine = multipleLogisticRegression(X_wine, y_wine, w_wine)
    print('Accuracy for iris data: %f' %(accuracy_iris))
    print('Accuracy for wine data: %f' %(accuracy_wine))
    return

def multipleIRLS(X, y, max_iter=20, tol=0.0001):
    N, p = X.shape
    K = y.shape[1]
    w = np.zeros((p,K))
    for _ in range(max_iter):
        S = _softmax(X, w)
        D = np.multiply(S, 1-S)
        YS = y - S
        for k in range(K):
            D_kk = np.diag(D[:, k])
            H = -np.matmul(X.T, np.matmul(D_kk, X))
            dw = np.matmul(X.T, YS[:, k])
            w[:, k] = w[:, k] - np.matmul(np.linalg.inv(H), dw)
    return w

def multipleLogisticRegression(X, y, w):
    y_hat = np.argmax(_softmax(X, w), axis=1)
    return np.mean(y_hat==np.argmax(y, axis=1))

# Some trick to avoid overflow
def _softmax(X, w):
    N, p = X.shape
    XW = np.matmul(X, w)
    ZW = XW - np.amax(XW, axis=1).reshape((N,1))
    # exp_XW = np.exp(np.matmul(X, w))
    exp_ZW = np.exp(ZW)
    # return exp_XW / np.sum(exp_XW, axis=1).reshape((N,1))
    return exp_ZW / np.sum(exp_ZW, axis=1).reshape((N,1))

def one_hot(y, num_classes):
    N = y.shape[0]
    result = np.zeros((N, num_classes))
    result[np.arange(N), y] = 1
    return result

if __name__ == '__main__':
    main()

