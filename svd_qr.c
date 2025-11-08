#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 2  // Example size; can be changed

void QR_Decomposition(double A[N][N], double Q[N][N], double R[N][N], size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            R[i][j] = 0.0;

    // Gram-Schmidt
    for (size_t i = 0; i < n; i++) {
        // Copy A[:, i] into u
        double u[N];
        for (size_t k = 0; k < n; k++)
            u[k] = A[k][i];

        // Subtract projections on previous Q columns
        for (size_t j = 0; j < i; j++) {
            double dot = 0.0;
            for (size_t k = 0; k < n; k++)
                dot += Q[k][j] * A[k][i];
            R[j][i] = dot;

            for (size_t k = 0; k < n; k++)
                u[k] -= R[j][i] * Q[k][j];
        }

        // Compute R[i][i] = ||u||
        double norm = 0.0;
        for (size_t k = 0; k < n; k++)
            norm += u[k] * u[k];
        norm = sqrt(norm);
        R[i][i] = norm;

        // Normalize Q[:, i]
        for (size_t k = 0; k < n; k++)
            Q[k][i] = u[k] / norm;
    }
}


void QR_SVD(double A[][N]){
    double AT[N][N] = {0};
    double AAt[N][N] = {0};
    double AtA[N][N] = {0};
    double U[N][N];
    double Utemp[N][N] = {0};
    double V[N][N];
    double Vtemp[N][N] = {0};
    double Q[N][N] = {0};
    double R[N][N] = {0};
    int iterations = 100;
    double eigvals[N][N] = {0};

    // Compute A transposed 
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            AT[j][i] = A[i][j];
        }
    }

    // Compute A @ A.T
    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++){
            AAt[i][j] = 0.0;
        }
    }
    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++){
            for (size_t k = 0; k < N; k++){
                AAt[i][j] += A[i][k] * AT[k][j];
            }
        }
    }

    // Compute A.T @ A
    for (size_t i = 0; i < N; i++){
        for(size_t j = 0; j < N; j++){
            AtA[i][j] = 0;
        }
    }
    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++){
            for (size_t k = 0; k < N; k++){
                AtA[i][j] += AT[i][k] * A[k][j];
            }
        }
    }

    // Initialize U, V as identity matrices NxN
    for (size_t i = 0; i < N; i++){
        for(size_t j = 0; j < N; j++){
            if(i == j){
                U[i][j] = 1;
                V[i][j] = 1;
            } else {
                U[i][j] = 0;
                V[i][j] = 0;
            }
        }
    }

    // Compute AAt eigenvector and eigenvalues via QR Decomposition
    for(int iter = 0; iter < iterations; iter++){
        // Step 1: QR decomposition
        QR_Decomposition(AAt, Q, R, N);

        // Step 2: New A = R @ Q
        double Anew[N][N] = {0};
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                for (size_t k = 0; k < N; k++){
                    Anew[i][j] += R[i][k] * Q[k][j];
                }
            }
        }

        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                AAt[i][j] = Anew[i][j];
            }
        }

        // Step 3: accumulate eigenvectors: U = U * Q
        double Utemp[N][N] = {0};
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                for (size_t k = 0; k < N; k++){
                    Utemp[i][j] += U[i][k] * Q[k][j];
                }
            }
        }
        // Copy Utemp into U
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                U[i][j] = Utemp[i][j];
            }
        }
    }
    // At the end of the iterations, take the resulting eigenvalues
    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++){
            for (size_t k = 0; k < N; k++){
                eigvals[i][j] += R[i][k] * Q[k][j];
            }
        }
    }


    // Compute AtA eigenvector
    for(int iter = 0; iter < iterations; iter++){
        // Step 1: QR decomposition
        QR_Decomposition(AtA, Q, R, N);

        // Step 2: New A = R @ Q
        double Anew[N][N] = {0};
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                for (size_t k = 0; k < N; k++){
                    Anew[i][j] += R[i][k] * Q[k][j];
                }
            }
        }

        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                AtA[i][j] = Anew[i][j];
            }
        }

        // Step 3: accumulate eigenvectors: V = V * Q
        double Vtemp[N][N] = {0};
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                for (size_t k = 0; k < N; k++){
                    Vtemp[i][j] += V[i][k] * Q[k][j];
                }
            }
        }
        // Copy Vtemp into V
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                V[i][j] = Vtemp[i][j];
            }
        }
    }

    printf("Eigenvalues:");
    for (size_t i = 0; i < N; i++){
        printf("\n");
        for (size_t j = 0; j < N; j++){
            if(i == j) printf("%f   ", eigvals[i][j]);
        }
    }

    printf("\n\nLeft singular values:");
    for (size_t i = 0; i < N; i++){
        printf("\n");
        for (size_t j = 0; j < N; j++){
            printf("%f  ", U[i][j]);
        }
    }

    printf("\n\nRight singular values:");
    for (size_t i = 0; i < N; i++){
        printf("\n");
        for (size_t j = 0; j < N; j++){
            printf("%f  ", V[i][j]);
        }
    }

    
}

int main(){
    double A[N][N] = {
        {1, 2},
        {2, 1}
    };
    QR_SVD(A);
    return 0;
}

