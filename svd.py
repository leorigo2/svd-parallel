import numpy as np

A = np.array([[1, 2, 1], [2, 1, 4], [3, 10, 1], [1, 2, 0]])
np.set_printoptions(precision=2)

print(np.linalg.eig(A.dot(A.T)))

# Numpy SVD for comparison
U_svd, S_svd, Vt_svd = np.linalg.svd(A, full_matrices = False)

print("-------------Numpy SVD-------------")

print("Left Singular Vectors are: \n", U_svd, "\n")
print("Sigular Values are: \n", S_svd, "\n")
print("Right Singular Vectors are: \n", Vt_svd)

