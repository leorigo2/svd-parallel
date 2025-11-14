#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define N 3  // columns    
#define M 4 // rows
#define min(a, b) ((a) < (b) ? (a) : (b))

int count = 3; // elements per cluster

int comm_sz; 
int my_rank;

MPI_Init(NULL, NULL);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

void QR_Decomposition(size_t n, double A[][n], double Q[][n], double R[][n]) {

    double *sub_r = (double *)malloc(sizeof(double) * count);

    for(int i = 0; i < M; i++) // one row per process
        MPI_Scatter(R[i], count, MPI_DOUBLE, sub_r, count, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    
    for (size_t i = 0; i < count; i++)
            sub_r[i] = 0.0;
    
    for(int i = 0; i < M; i++)
        MPI_Gather(sub_r, count, MPI_DOUBLE, R[i], count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(sub_r);

    MPI_Finalize();

    // Gram-Schmidt
    for (size_t i = 0; i < n; i++) {
        // Copy A[:, i] into u
        double u[n];
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
        
        norm = norm;
        R[i][i] = norm;

        // Normalize Q[:, i]
        for (size_t k = 0; k < n; k++){
            if(norm == 0) continue; 
            Q[k][i] = u[k] / norm;
        }
    }
}


void QR_SVD(double A[][N]){
    double Anew[M][M] = {0};
    double AT[N][M] = {0};
    double AAt[M][M] = {0};
    double AtA[N][N] = {0};
    double U[M][M];
    double Utemp[M][M] = {0};
    double V[N][N];
    double Vtemp[N][N] = {0};
    double Q_AAt[M][M] = {0};
    double Q_AtA[N][N] = {0};
    double R_AAt[M][M] = {0};
    double R_AtA[N][N] = {0};
    int iterations = 10;
    double eigvals[N][N] = {0};

    // Compute A transposed 
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            AT[j][i] = A[i][j];
        }
    }

    // Compute A @ A.T
    for (size_t i = 0; i < M; i++){
        for (size_t j = 0; j < M; j++){
            AAt[i][j] = 0.0;
        }
    }
    for (size_t i = 0; i < M; i++){
        for (size_t j = 0; j < M; j++){
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
            for (size_t k = 0; k < M; k++){
                AtA[i][j] += AT[i][k] * A[k][j];
            }
        }
    }

    // Initialize U, V as identity matrices NxN
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            U[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }


    // Compute AAt eigenvector and eigenvalues via QR Decomposition
    for(int iter = 0; iter < iterations; iter++){
        // Step 1: QR decomposition
        QR_Decomposition(M, AAt, Q_AAt, R_AAt);

        // Step 2: New A = R @ Q
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<M;j++)
                Anew[i][j] = 0.0;

        for (size_t i = 0; i < M; i++){
            for (size_t j = 0; j < M; j++){
                for (size_t k = 0; k < M; k++){
                    Anew[i][j] += R_AAt[i][k] * Q_AAt[k][j];
                }
            }
        }

        for (size_t i = 0; i < M; i++){
            for (size_t j = 0; j < M; j++){
                AAt[i][j] = Anew[i][j];
            }
        }

        // Step 3: accumulate eigenvectors: U = U * Q
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<M;j++)
                Utemp[i][j] = 0.0;
        for (size_t i = 0; i < M; i++){
            for (size_t j = 0; j < M; j++){
                for (size_t k = 0; k < M; k++){
                    Utemp[i][j] += U[i][k] * Q_AAt[k][j];
                }
            }
        }
        // Copy Utemp into U
        for (size_t i = 0; i < M; i++){
            for (size_t j = 0; j < M; j++){
                U[i][j] = Utemp[i][j];
            }
        }
    }
    for (size_t i = 0; i < M; i++)
        eigvals[i][i] = AAt[i][i];


    // Compute AtA eigenvector
    for(int iter = 0; iter < iterations; iter++){
        // Step 1: QR decomposition
        QR_Decomposition(N, AtA, Q_AtA, R_AtA);

        // Step 2: New A = R @ Q
        for(size_t i=0;i<N;i++)
            for(size_t j=0;j<N;j++)
                Anew[i][j] = 0.0;
                
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                for (size_t k = 0; k < N; k++){
                    Anew[i][j] += R_AtA[i][k] * Q_AtA[k][j];
                }
            }
        }

        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                AtA[i][j] = Anew[i][j];
            }
        }

        // Step 3: accumulate eigenvectors: V = V * Q
        for(size_t i=0;i<N;i++)
            for(size_t j=0;j<N;j++)
                Vtemp[i][j] = 0.0;
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                for (size_t k = 0; k < N; k++){
                    Vtemp[i][j] += V[i][k] * Q_AtA[k][j];
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
    int rank = min(N, M);
    printf("Eigenvalues:");
    for (size_t i = 0; i < N; i++){
        printf("\n");
        for (size_t j = 0; j < N; j++){
            if(i == j) printf("%f   ", eigvals[i][j]);
        }
    }

    printf("\n\nLeft singular values:");
    for (size_t i = 0; i < M; i++){
        printf("\n");
        for (size_t j = 0; j < rank; j++){
            printf("%f  ", U[i][j]);
        }
    }

    printf("\n\nRight singular values:");
    for (size_t i = 0; i < rank; i++){
        printf("\n");
        for (size_t j = 0; j < N; j++){
            printf("%f  ", V[j][i]);
        }
    }

    
}

int main(){
    printf("here");
    double A[M][N] = {
        {1, 2, 1},
        {2, 1, 4},
        {3, 10, 1},
        {1, 2, 0}
    };
   
    QR_SVD(A);
    return 0;
}

