import numpy as np
import time
import math


def power_svd(A, iters, epsilon = 1e-10):
    mu, sigma = 0, 1
    n, m = A.shape
    x = np.random.normal(mu, sigma, size=min(n, m))
    last_x = None
    current_x = x
    n, m = A.shape
    if n > m:
        B = A.T.dot(A)
    else:
        B = A.dot(A.T)
    for i in range(iters):
        last_x = current_x
        current_x = B.dot(last_x)
        current_x /= np.linalg.norm(current_x)


    if n > m:
        v = current_x
        sigma = np.linalg.norm(A.dot(v))
        u = A.dot(v) / sigma
    else:
        u = current_x
        sigma = np.linalg.norm(A.T.dot(u))
        v = A.T.dot(u) / sigma
        
    return np.reshape(
        u, (A.shape[0], 1)), sigma, np.reshape(
        v, (A.shape[1], 1))


def main():
    A = np.array([[7, 2],
                  [2, 4],
                  [0, 1]], dtype=float)
    n, m = A.shape
    rank = min(n, m)
    U = np.zeros((A.shape[0], 1))
    S = []
    V = np.zeros((A.shape[1], 1))
    
    iterations = 100

    # SVD using Power Method
    for i in range(rank):
        u, sigma, v = power_svd(A, iterations)
        U = np.hstack((U, u))
        S.append(sigma)
        V = np.hstack((V, v))
        A = A - sigma * np.outer(u, v)

    
    print("-------------Power method-------------")
    print("Left Singular Vectors are: \n", U[:, 1:], "\n")
    print("Sigular Values are: \n", S, "\n")
    print("Right Singular Vectors are: \n", V[:, 1:].T)

    A = U[:, 1:] @ np.diag(S) @ V[:, 1:].T
    print("\nOriginal matrix reconstructed: \n", A)


if __name__ == '__main__':
    main()
