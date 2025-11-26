import numpy as np
import math

def QR_Decomposition(A):
    n, m = A.shape # get the shape of A

    Q = np.empty((n, n)) # initialize matrix Q
    u = np.empty((n, n)) # initialize matrix u

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # get each u vector
            print("dot: ", (A[:, i] @ Q[:, j]))
            print("u: ", u[:, i], "\n")

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R

def main():
    A = np.array([[1, 2, 1], [2, 1, 4], [3, 10, 1]])
    AAt = A.dot(A.T)
    AtA = A.T.dot(A)
    U = np.eye(A.shape[0])
    V = np.eye(A.shape[1])

    # Define the number of iterations ???
    delta = 0.001
    epsilon = 0.99
    lamda = 1
    iterations = int(math.log(
        4 * math.log(2 * A.shape[1] / delta) / (epsilon * delta)) / (2 * lamda))
    
    iterations = 10

    # SVD using QR method

    # Start with AAt (U matrix and eigenvalues) 
    eigvals = AAt.copy()
    for i in range(iterations):
        Q, R = QR_Decomposition(eigvals)
        U = U @ Q
        eigvals = R @ Q

    # Now AtA (V matrix)
    for i in range(iterations):
        Q, R = QR_Decomposition(AtA)
        V = V @ Q
        AtA = R @ Q


    print("-------------QR method-------------")
    print("Left Singular Vectors are: \n", U, "\n")
    print("Sigular Values are: \n", np.sqrt(np.diag(eigvals)), "\n")
    print("Right Singular Vectors are: \n", V)


if __name__ == '__main__':
    main()
