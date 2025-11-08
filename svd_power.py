import numpy as np
import time
import math


def power_svd(A, iters):
    mu, sigma = 0, 1
    x = np.random.normal(mu, sigma, size=A.shape[1])
    B = A.T.dot(A)
    for i in range(iters):
        new_x = B.dot(x)
        x = new_x
    v = x / np.linalg.norm(x)
    sigma = np.linalg.norm(A.dot(v))
    u = A.dot(v) / sigma
    return np.reshape(
        u, (A.shape[0], 1)), sigma, np.reshape(
        v, (A.shape[1], 1))


def main():
    A = np.array([[1, 0, 2], [3, 4, 0], [1, 2, 1]])
    rank = np.linalg.matrix_rank(A)
    U = np.zeros((A.shape[0], 1))
    S = []
    V = np.zeros((A.shape[1], 1))

    # Define the number of iterations ???
    delta = 0.001
    epsilon = 0.99
    lamda = 1
    iterations = int(math.log(
        4 * math.log(2 * A.shape[1] / delta) / (epsilon * delta)) / (2 * lamda))
    
    iterations = 100

    # SVD using Power Method
    for i in range(rank):
        u, sigma, v = power_svd(A, iterations)
        U = np.hstack((U, u))
        S.append(sigma)
        V = np.hstack((V, v))
        A = A - u.dot(v.T).dot(sigma)

    
    print("-------------Power method-------------")
    print("Left Singular Vectors are: \n", U[:, 1:], "\n")
    print("Sigular Values are: \n", S, "\n")
    print("Right Singular Vectors are: \n", V[:, 1:])


if __name__ == '__main__':
    main()
