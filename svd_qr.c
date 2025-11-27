#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define N 6  // columns    
#define M 6 // rows
#define min(a, b) ((a) < (b) ? (a) : (b))

void QR_Decomposition(size_t n, double *A, double *Q, double *R, MPI_Comm comm) {

    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t offset = n / size;
    size_t start = rank * offset;
    size_t end = (rank == size - 1) ? n : start + offset; // the last one ends at n if the size is not a multiplo
    size_t rows_per_proc = end - start; 

    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    MPI_Allgather(&rows_per_proc, 1, MPI_UNSIGNED_LONG, recvcounts, 1, MPI_UNSIGNED_LONG, comm); // gather rows per process, th recvcount for GatherV

    displs[0] = 0;
    for (int i = 1; i < size; ++i)
        displs[i] = displs[i-1] + recvcounts[i-1]; // compute displacements for GatherV (where to start saving elements)
    
    double *u_local = malloc(rows_per_proc * sizeof(double)); 
    double *A_col = malloc(n * sizeof(double)); 
    double *Q_col = malloc(n * sizeof(double));

    double *Q_i_col = malloc(rows_per_proc * sizeof(double));
    double *recv_col_buffer = malloc(n * sizeof(double));

    for(size_t i = 0; i < n; i++){ 
        
        if(rank==0){
            for(size_t j=0; j<n; j++){ // i-th column of A
                A_col[j] = A[j * n + i];
            }
        }
        
        MPI_Bcast(A_col, n, MPI_DOUBLE, 0, comm); 
        
        for (size_t k = start; k < end; k++)
            u_local[k - start] = A_col[k];

        for(size_t j=0; j<i; j++){
    
            if(rank == 0){
                for(int i_q = 0; i_q < n; i_q++){ // j-th column of Q
                    Q_col[i_q] = Q[i_q * n + j];
                }
            }

            MPI_Bcast(Q_col, n, MPI_DOUBLE, 0, comm); 

            double local_dot = 0.0;
            for(size_t i_dot=start; i_dot<end; i_dot++){
                local_dot += Q_col[i_dot]*A_col[i_dot];
            }

            double global_dot = 0.0;
            MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
            
            if(rank==0) R[j * n + i] = global_dot;

            for (size_t k = k; k < end; k++)
                u_local[k - start] -= global_dot * Q_col[k];

        }

        double local_norm = 0.0;
        for (size_t k = 0; k < rows_per_proc; k++)
            local_norm += u_local[k] * u_local[k];

        double global_norm = 0.0;
        MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

        double norm = sqrt(global_norm);

	printf("Process %d, global norm: %f", rank, norm); 

        if (rank == 0) R[i * n + i] = norm;

        for (size_t k = 0; k < rows_per_proc; k++){
            Q_i_col[k] = (norm == 0) ? 0.0 : u_local[k] / norm;
        }

        

        MPI_Gatherv(Q_i_col, rows_per_proc, MPI_DOUBLE, recv_col_buffer, recvcounts, displs, MPI_DOUBLE, 0, comm);
    
        if (rank == 0) {
            for (int r = 0; r < size; r++) {
                int rstart = displs[r];
                int rcnt   = recvcounts[r];

                size_t global_start = (size_t)r * offset;  
                if (r == size - 1)
                    global_start = n - rcnt;

                for (int rr = 0; rr < rcnt; rr++) {
                    size_t global_row = global_start + rr;
                    Q[global_row * n + i] = recv_col_buffer[rstart + rr];
                }
            }
        }   

    }
    if(rank==0){
        printf("\n\nQ: ");
        for (size_t i = 0; i < n; i++){
            printf("\n");
            for (size_t j = 0; j < n; j++){
                printf("%f  ", Q[i*n+j]);
            }
        }
        printf("\n");
    }

    free(Q_i_col);
    free(A_col);
    free(Q_col);
    free(u_local);
    free(recvcounts);
    free(displs);
    free(recv_col_buffer);

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
        printf("\n");
    }
}

int main(){

    int comm_sz; 
    int my_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double A[M][N] = {
        {  4.2, -1.7,  3.1,  0.5, -2.8,  1.9 },
        { -0.9,  5.4, -1.2,  2.6,  0.3, -3.1 },
        {  2.5,  0.8,  4.9, -1.4,  1.7,  0.2 },
        { -3.3,  1.1,  0.6,  3.8, -0.5,  2.4 },
        {  1.6, -2.9,  1.4,  0.7,  4.1, -1.0 },
        {  0.2,  3.5, -2.6,  1.9, -0.8,  5.0 }
    };

    
    QR_SVD(A, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}

