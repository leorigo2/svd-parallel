#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

double** read_matrix(FILE* file, int R, int C){ // R rows of matrix, C columns of matrix
    double** matrix = (double**)malloc(R * sizeof(double*)); // array of R pointers
    for (int i = 0; i < R; i++) {
        matrix[i] = (double*)malloc(C * sizeof(double)); // each rows has C elements
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            fscanf(file, "%lf", &matrix[i][j]);
        }
    }
    return matrix;
}

double** alloc_matrix(int M, int N) {
    double** matrix = (double**)malloc(M * sizeof(double*));
    for (int i = 0; i < M; i++) {
        // Use calloc to allocate and initialize all elements to zero
        matrix[i] = (double*)calloc(N, sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int R) {
    if (matrix != NULL) {
        for (int i = 0; i < R; i++) {
            free(matrix[i]);
        }
        free(matrix);
    }
}

void matrix_multiplication(size_t m, size_t n, double** A, double** B, double** C){ // m rows of A, n column of A

    int i, j, k; 
    int threads = omp_get_max_threads();
    printf("\nthreads: %d", threads);
    # pragma omp parallel for num_threads(threads) private(i, j, k)
    for (i = 0; i < m; ++i) {
        for (j = 0; j < m; ++j) {
            for (k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

}

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

    int local_rows_int = (int) rows_per_proc;

    MPI_Allgather(&local_rows_int, 1, MPI_INT, recvcounts, 1, MPI_INT, comm); // gather rows per process, th recvcount for GatherV

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

            for (size_t k = start; k < end; k++)
                u_local[k - start] -= global_dot * Q_col[k];

        }

        double local_norm = 0.0;
        for (size_t k = 0; k < rows_per_proc; k++)
            local_norm += u_local[k] * u_local[k];

        double global_norm = 0.0;
        MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

        double norm = sqrt(global_norm);

        if (rank == 0) R[i * n + i] = norm;

        for (size_t k = 0; k < rows_per_proc; k++){
            Q_i_col[k] = (norm == 0) ? 0.0 : u_local[k] / norm;
        }

        MPI_Gatherv(Q_i_col, rows_per_proc, MPI_DOUBLE, recv_col_buffer, recvcounts, displs, MPI_DOUBLE, 0, comm);

        if (rank == 0) {
            for (int r = 0; r < size; r++) {
                int displacement = displs[r]; // displacement of Rth node
                int elements_count = recvcounts[r]; // number of elements of Rth node

                size_t global_start = (size_t)r * offset; // starting point of Rth node
                if (r == size - 1)
                    global_start = n - elements_count; // if it's the last node, subtract the remaining elements (it could be not a multiplo)

                for (int rr = 0; rr < elements_count; rr++) {
                    size_t global_row = global_start + rr; // compute the row for each element
                    Q[global_row * n + i] = recv_col_buffer[displacement + rr]; // put each element in Q[row][i] so i-th column of Q
                }
            }
        }   

    }

    free(Q_i_col);
    free(A_col);
    free(Q_col);
    free(u_local);
    free(recvcounts);
    free(displs);
    free(recv_col_buffer);

}


void QR_SVD(double** A, int M, int N, MPI_Comm comm){

    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double** AT = alloc_matrix(N, M); // A^T (N x M)
    double** AAt = alloc_matrix(M, M); // A A^T (M x M)
    double** AtA = alloc_matrix(N, N); // A^T A (N x N)
    double** U = alloc_matrix(M, M); // Left Singular Vectors
    double** Utemp = alloc_matrix(M, M);
    double** V = alloc_matrix(N, N); // Right Singular Vectors
    double** Vtemp = alloc_matrix(N, N);
    double** Q_AAt = alloc_matrix(M, M); 
    double** R_AAt = alloc_matrix(M, M); 
    double** Q_AtA = alloc_matrix(N, N);
    double** R_AtA = alloc_matrix(N, N);
    double** eigvals = alloc_matrix(N, N); 

    int iterations = 10; 

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
        matrix_multiplication(M, N, A, AT, AAt);

        // Compute A.T @ A
        matrix_multiplication(N, M, AT, A, AtA);

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

	    double** Anew = alloc_matrix(M, M);
        // Step 2: New A = R @ Q
        if(rank == 0){
            matrix_multiplication(M, M, R_AAt, Q_AAt, Anew);

            for (size_t i = 0; i < M; i++){
                for (size_t j = 0; j < M; j++){
                    AAt[i][j] = Anew[i][j];
                }
            }

            // Step 3: accumulate eigenvectors: U = U * Q
            for(size_t i=0;i<M;i++)
                for(size_t j=0;j<M;j++)
                    Utemp[i][j] = 0.0;

            matrix_multiplication(M, M, U, Q_AAt, Utemp);

            // Copy Utemp into U
            for (size_t i = 0; i < M; i++){
                for (size_t j = 0; j < M; j++){
                    U[i][j] = Utemp[i][j];
                }
            }
        }
        free_matrix(Anew, M);
    }

    if(rank == 0){
        for (size_t i = 0; i < M; i++)
            eigvals[i][i] = AAt[i][i];
    }


    // Compute AtA eigenvector
    for(int iter = 0; iter < iterations; iter++){
        // Step 1: QR decomposition
        QR_Decomposition(N, (double *)AtA, (double *)Q_AtA, (double *)R_AtA, comm);

	    double** Anew = alloc_matrix(N, N);
        // Step 2: New A = R @ Q
        if(rank == 0){
            matrix_multiplication(N, N, R_AtA, Q_AtA, Anew);

            for (size_t i = 0; i < N; i++){
                for (size_t j = 0; j < N; j++){
                    AtA[i][j] = Anew[i][j];
                }
            }

            // Step 3: accumulate eigenvectors: V = V * Q
            for(size_t i=0;i<N;i++)
                for(size_t j=0;j<N;j++)
                    Vtemp[i][j] = 0.0;

           matrix_multiplication(N, N, V, Q_AtA, Vtemp);

            // Copy Vtemp into V
            for (size_t i = 0; i < N; i++){
                for (size_t j = 0; j < N; j++){
                    V[i][j] = Vtemp[i][j];
                }
            }
        }
        free_matrix(Anew, N);
    }

    if(rank == 0){
        int mat_rank = min(N, M);
        printf("Eigenvalues:\n");
        for (size_t i = 0; i < N; i++){
            printf("\n");
            for (size_t j = 0; j < N; j++){
                if(i==j) printf("%f   ", sqrt(eigvals[i][j]));
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

        fflush(stdout);
    }
    
    free_matrix(AT, N);
    free_matrix(AAt, M);
    free_matrix(AtA, N);
    free_matrix(U, M);
    free_matrix(Utemp, M);
    free_matrix(V, N);
    free_matrix(Vtemp, N);
    free_matrix(Q_AAt, M);
    free_matrix(R_AAt, M);
    free_matrix(Q_AtA, N);
    free_matrix(R_AtA, N);
    free(eigvals);
}

int main(int argc, char* argv[]){

    int comm_sz; 
    int my_rank;
    double start_time, end_time, elapsed_time; 
    int num_matrices = 0;

    int R = 0, C = 0;
    double** current_matrix = NULL;
    int elements = 0;

    FILE* dataset = NULL;
    FILE* results = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0){
        dataset = fopen("dataset.txt", "r");
        results = fopen("results_parallel.txt", "w");

        fprintf(results, "elements time");
    
        fscanf(dataset, "%d", &num_matrices); // read the number of matrices in the dataset
    }

    MPI_Bcast(&num_matrices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for(int k = 0; k < num_matrices; k++){
        if(my_rank == 0){
            fscanf(dataset, "%d %d", &R, &C); // read number of rows and columns
        }

        MPI_Bcast(&R, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
        elements = R*C;

        if(my_rank == 0){
            current_matrix = read_matrix(dataset, R, C);
        }

        MPI_Barrier(MPI_COMM_WORLD); // Start all processes
        start_time = MPI_Wtime();

        QR_SVD(current_matrix, R, C, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD); // Wait all processes to finish
        end_time = MPI_Wtime();

        elapsed_time = end_time - start_time;
        
        if (my_rank == 0) {
            fprintf(results, "%d %f\n", elements, elapsed_time);
        }

        free_matrix(current_matrix, R);

    }

    if(my_rank == 0){
        fclose(dataset);
        fclose(results);
    }
    
    MPI_Finalize();
    return 0;
}
