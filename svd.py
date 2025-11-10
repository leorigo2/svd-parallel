import numpy as np

A =np.array([[1, 2, 1],
                  [2, 3, 0]], dtype=float)

np.set_printoptions(precision=2)

# Numpy SVD for comparison
U_svd, S_svd, Vt_svd = np.linalg.svd(A)

print("-------------Numpy SVD-------------")

print("Left Singular Vectors are: \n", U_svd, "\n")
print("Sigular Values are: \n", S_svd, "\n")
print("Right Singular Vectors are: \n", Vt_svd)

