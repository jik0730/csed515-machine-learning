import sys
import numpy as np
from sklearn.preprocessing import normalize

def main():
    # Iris data
    X_iris = np.genfromtxt('./data/iris.data', delimiter=',', usecols=(0,1,2,3), dtype=float)[:100, :]
    y_iris = np.genfromtxt('./data/iris.data', delimiter=',', usecols=(4), dtype=str)[:100]
    y_iris[y_iris=='Iris-setosa'] = 0
    y_iris[y_iris=='Iris-versicolor'] = 1
    y_iris = y_iris.astype(int)
    X_iris = normalize(X_iris)


    # Wine data
    X_wine = np.genfromtxt('./data/wine.data', delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13), dtype=float)[:130, :]
    y_wine = np.genfromtxt('./data/wine.data', delimiter=',', usecols=(0), dtype=int)[:130]
    X_wine = normalize(X_wine)
    y_wine -= 1

    # IRLS for binary case
    w_iris = binaryIRLS(X_iris, y_iris)
    w_wine = binaryIRLS(X_wine, y_wine)

    # Evaluate accuracy of logistic regression
    accuracy_iris = binaryLogisticRegression(X_iris, y_iris, w_iris)
    accuracy_wine = binaryLogisticRegression(X_wine, y_wine, w_wine)
    print('Accuracy for iris data: %f' %(accuracy_iris))
    print('Accuracy for wine data: %f' %(accuracy_wine))
    return

def binaryIRLS(X, y, max_iter=20, tol=0.0001):
    # Vectorized operation
    N, p = X.shape
    w = np.random.rand(p, N)**2*0.01
    w = np.zeros((p,N))
    for _ in range(max_iter):
        _w = np.sum(w, axis=1)
        V_r = np.diag(np.array([_sigma(_w, X[i])*(1-_sigma(_w, X[i])) for i in range(N)]))
        VY = np.diag(np.array([y[i]-_sigma(_w, X[i]) for i in range(N)]))
        w = w + np.dot(np.linalg.inv(X.T.dot(V_r).dot(X)), X.T.dot(VY))
        __w = np.sum(w, axis=1)
        if np.sum(np.abs(_w-__w)) < tol:
            return __w
    return __w

    # None vectorized operation
    # w = np.zeros(p)
    # for _ in range(max_iter):
    #     H = np.zeros((p,p))
    #     G = np.zeros((p,1))
    #     for i in range(N):
    #         coefficient = _sigma(w, X[i])*(1-_sigma(w, X[i]))
    #         coefficient2 = y[i] - _sigma(w, X[i])
    #         H += coefficient * np.matmul(X[i].reshape((p,1)), X[i].reshape((1,p)))
    #         G += coefficient2 * X[i].reshape((p,1))
    #     print(G.shape)
    #     print(H.shape)
    #     w = w + np.matmul(np.linalg.inv(H), G).reshape(p)
    # return w

def binaryLogisticRegression(X, y, w):
    N, p = X.shape
    _y = np.array([_sigma(w, X[i]) for i in range(N)])
    _y[_y>=0.5] = 1
    _y[_y<0.5] = 0
    return np.mean(y==_y)

def _sigma(w, x):
    return float(1) / (float(1) + np.exp(-w.dot(x)))

if __name__ == '__main__':
    main()

