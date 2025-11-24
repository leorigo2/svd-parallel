#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define N 3  // columns    
#define M 3 // rows
#define min(a, b) ((a) < (b) ? (a) : (b))

void QR_Decomposition(size_t n, double *A, double *Q, double *R, MPI_Comm comm) {

    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t rows_per_proc = n / size;
    size_t start = rank * rows_per_proc;
    size_t end = (rank == size - 1) ? n : start + rows_per_proc; // the last one ends at n if the size is not a multiplo

    double *u_local = malloc((end - start) * sizeof(double)); 
    double *A_col = malloc(n * sizeof(double)); 
    double *Q_col = malloc(n * sizeof(double));


    for(size_t i = 0; i < n; i++){ // some rows per process
        
        if(rank==0){
            for(size_t j=0; j<n; j++){ // i-th row of A
                A_col[j] = A[i * n + j];
            }
        }
        
        MPI_Bcast(A_col, n, MPI_DOUBLE, 0, comm); 
        
        for (size_t k = start; k < end; k++)
            u_local[k - start] = A_col[k];

        for(size_t j=0; j<n; j++){
            double local_dot = 0.0;
            if(rank == 0){
                for(int i_q = 0; i_q < n; i_q++){ // j-th row of Q
                    Q_col[i_q] = Q[j * n + i_q];
                }
            }

            MPI_Bcast(Q_col, n, MPI_DOUBLE, 0, comm); 
    
            for(size_t i_dot=start; i_dot<end; i_dot++){
                local_dot += Q_col[i_dot]*A_col[i_dot];
            }

            double global_dot = 0.0;
            MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
            
            R[i * n + j] = global_dot;

            for (size_t k = start; k < end; k++)
                u_local[k - start] -= R[i * n + j] * Q_col[k];

        }

        double local_norm = 0.0;
        for (size_t k = 0; k < (end - start); k++)
            local_norm += u_local[k] * u_local[k];

        double global_norm = 0.0;
        MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

        double norm = global_norm;
        R[i * n + i] = norm;

        for (size_t k = start; k < end; k++)
            Q[i * n + k] = (norm == 0) ? 0.0 : u_local[k - start] / norm;
    }

    
    free(A_col);
    free(Q_col);
    free(u_local);

}


void QR_SVD(double A[][N], MPI_Comm comm){

    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double Anew[M][M] = {0.0};
    double AT[N][M] = {0.0};
    double AAt[M][M] = {0.0};
    double AtA[N][N] = {0.0};
    double U[M][M];
    double Utemp[M][M] = {0.0};
    double V[N][N];
    double Vtemp[N][N] = {0.0};
    double Q_AAt[M][M] = {0.0};
    double Q_AtA[N][N] = {0.0};
    double R_AAt[M][M] = {0.0};
    double R_AtA[N][N] = {0.0};
    int iterations = 10;
    double eigvals[N][N] = {0.0};

    if(rank == 0){
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
    }


    // Compute AAt eigenvector and eigenvalues via QR Decomposition
    for(int iter = 0; iter < iterations; iter++){
        // Step 1: QR decomposition
        QR_Decomposition(M, (double *)AAt, (double *)Q_AAt, (double *)R_AAt, comm);

        // Step 2: New A = R @ Q
        if(rank == 0){
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
    }

    if(rank == 0){
        for (size_t i = 0; i < M; i++)
            eigvals[i][i] = AAt[i][i];
    }


    // Compute AtA eigenvector
    for(int iter = 0; iter < iterations; iter++){
        // Step 1: QR decomposition
        QR_Decomposition(N, (double *)AtA, (double *)Q_AtA, (double *)R_AtA, comm);

        // Step 2: New A = R @ Q
        if(rank == 0){
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
    }

    if(rank == 0){
        int mat_rank = min(N, M);
        printf("Eigenvalues:");
        for (size_t i = 0; i < N; i++){
            printf("\n");
            for (size_t j = 0; j < N; j++){
                printf("%f   ", eigvals[i][j]);
            }
        }

        printf("\n\nLeft singular values:");
        for (size_t i = 0; i < M; i++){
            printf("\n");
            for (size_t j = 0; j < mat_rank; j++){
                printf("%f  ", U[i][j]);
            }
        }

        printf("\n\nRight singular values:");
        for (size_t i = 0; i < mat_rank; i++){
            printf("\n");
            for (size_t j = 0; j < N; j++){
                printf("%f  ", V[j][i]);
            }
        }
    }
}

int main(){

    int comm_sz; 
    int my_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double A[M][N] = {
        {1, 2, 1},
        {2, 1, 4},
        {3, 10, 1}
    };
    
    QR_SVD(A, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}

