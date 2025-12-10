#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

void read_matrix(FILE* file, int R, int C, double** matrix){ // R rows of matrix, C columns of matrix
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            fscanf(file, "%lf", &matrix[i][j]);
        }
    }
}

double** alloc_matrix(int rows, int cols) {
    double **matrix = (double **)malloc(rows * sizeof(double*));

    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)calloc(cols, sizeof(double));
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
    for (i = 0; i < m; ++i) {
        for (j = 0; j < m; ++j) {
            for (k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

}

void parallel_matrix_multiplication(int m, int n, double** A, double** B, double** C, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double* flat_B = (double*)malloc(n * m * sizeof(double));
    
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                flat_B[i * m + j] = B[i][j];
            }
        }
    }

    MPI_Bcast(flat_B, n * m, MPI_DOUBLE, 0, comm);

    int* sendcounts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));

    int base_rows = m / size;
    int start = rank * base_rows;
    int end = (rank == size - 1) ? m : start + base_rows; 
    int my_rows = end - start; 

    for (int i = 0; i < size; i++) {
        int start_i = i * base_rows;
        int end_i = (i == size - 1) ? m : start_i + base_rows;
        int rows_i = end_i - start_i; 
        sendcounts[i] = rows_i * n; 
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
    }

    double* local_A = (double*)malloc(my_rows * n * sizeof(double));
    double* flat_A = NULL;

    if (rank == 0) {
        flat_A = (double*)malloc(m * n * sizeof(double));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                flat_A[i * n + j] = A[i][j];
            }
        }
    }

    MPI_Scatterv(flat_A, sendcounts, displs, MPI_DOUBLE, 
                 local_A, my_rows * n, MPI_DOUBLE, 
                 0, comm);

    double* local_C = (double*)calloc(my_rows * m, sizeof(double));

    for (int i = 0; i < my_rows; i++) {
        for (int k = 0; k < n; k++) {
            double a_val = local_A[i * n + k];
            for (int j = 0; j < m; j++) {
                local_C[i * m + j] += a_val * flat_B[k * m + j];
            }
        }
    }

 
    for (int ii = 0; ii < size; ii++) {
        int start_ii = ii * base_rows;
        int end_ii = (ii == size - 1) ? m : start_ii + base_rows;
        int rows_ii = end_ii - start_ii; 
        
        sendcounts[ii] = rows_ii * m; 
        displs[ii] = (ii == 0) ? 0 : displs[ii-1] + sendcounts[ii-1];
    }

    double* flat_C = (double*)malloc(m * m * sizeof(double));

    MPI_Allgatherv(local_C, my_rows * m, MPI_DOUBLE,
                flat_C, sendcounts, displs, MPI_DOUBLE,
                comm);


    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            C[i][j] = flat_C[i * m + j];
        }
    }
    if(rank==0) free(flat_A);

    free(flat_C);
    free(flat_B);
    free(local_A);
    free(local_C);
    free(sendcounts);
    free(displs);
}

void QR_Decomposition(size_t n, double **A, double **Q, double **R, MPI_Comm comm) {

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

        for (size_t k = start; k < end; k++)
            u_local[k - start] = A[k][i];

        for(size_t j=0; j<i; j++){

            double local_dot = 0.0;
            for(size_t i_dot=start; i_dot<end; i_dot++){
                local_dot += Q[i_dot][j]*A[i_dot][i];
            }

            double global_dot = 0.0;
            MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
            
            R[j][i] = global_dot;

            for (size_t k = start; k < end; k++)
                u_local[k - start] -= global_dot * Q[k][j];

        }

        double local_norm = 0.0;
        for (size_t k = 0; k < rows_per_proc; k++)
            local_norm += u_local[k] * u_local[k];

        double global_norm = 0.0;
        MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

        double norm = sqrt(global_norm);

        R[i][i] = norm;

        for (size_t k = 0; k < rows_per_proc; k++){
            Q_i_col[k] = (norm == 0) ? 0.0 : u_local[k] / norm;
        }

        for (size_t k = 0; k < rows_per_proc; k++) {
            size_t global_row = start + k;
            Q[global_row][i] = Q_i_col[k]; 
        }

        MPI_Allgatherv(Q_i_col, rows_per_proc, MPI_DOUBLE, recv_col_buffer, recvcounts, displs, MPI_DOUBLE, comm);


        for (int r = 0; r < size; r++) {
            int displacement = displs[r]; // displacement of Rth node
            int elements_count = recvcounts[r]; // number of elements of Rth node

            size_t global_start = (size_t)r * offset; // starting point of Rth node
            if (r == size - 1)
                global_start = n - elements_count; // if it's the last node, subtract the remaining elements (it could be not a multiplo)

            for (int rr = 0; rr < elements_count; rr++) {
                size_t global_row = global_start + rr; // compute the row for each element
                Q[global_row][i] = recv_col_buffer[displacement + rr]; // put each element in Q[row][i] so i-th column of Q
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
    double start_time, end_time; // Variables for timing

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
    double** eigvals = alloc_matrix(M, M); 

    int iterations = 10; 


    // Compute A transposed 
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            AT[j][i] = A[i][j];
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

    // --- Compute A @ A.T ---
    if (rank == 0) start_time = MPI_Wtime();
    parallel_matrix_multiplication(M, N, A, AT, AAt, comm);
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("[Time] A @ A.T multiplication: %f seconds\n", end_time - start_time);
    }

    // --- Compute A.T @ A ---
    if (rank == 0) start_time = MPI_Wtime();
    parallel_matrix_multiplication(N, M, AT, A, AtA, comm);
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("[Time] A.T @ A multiplication: %f seconds\n", end_time - start_time);
    }
   
    // Compute AAt eigenvector and eigenvalues via QR Decomposition
    for(int iter = 0; iter < iterations; iter++){
        
        // Step 1: QR decomposition
        if (rank == 0) start_time = MPI_Wtime();
        QR_Decomposition(M, AAt, Q_AAt, R_AAt, comm);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("[Time] Iter %d (AAt): QR_Decomposition: %f seconds\n", iter, end_time - start_time);
        }

        double** Anew = alloc_matrix(M, M);
        
        // Step 2: New A = R @ Q
        if (rank == 0) start_time = MPI_Wtime();
        parallel_matrix_multiplication(M, M, R_AAt, Q_AAt, Anew, comm);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("[Time] Iter %d (AAt): MatMul (R @ Q): %f seconds\n", iter, end_time - start_time);
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
    

        if (rank == 0) start_time = MPI_Wtime();
        parallel_matrix_multiplication(M, M, U, Q_AAt, Utemp, comm);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("[Time] Iter %d (AAt): MatMul (Accumulate U): %f seconds\n", iter, end_time - start_time);
        }

        // Copy Utemp into U
        for (size_t i = 0; i < M; i++){
            for (size_t j = 0; j < M; j++){
                U[i][j] = Utemp[i][j];
            }
        }
    
        free_matrix(Anew, M);
    }


    for (size_t i = 0; i < M; i++)
        eigvals[i][i] = AAt[i][i];
    


    // Compute AtA eigenvector
    for(int iter = 0; iter < iterations; iter++){
        
        // Step 1: QR decomposition
        if (rank == 0) start_time = MPI_Wtime();
        QR_Decomposition(N, AtA, Q_AtA, R_AtA, comm);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("[Time] Iter %d (AtA): QR_Decomposition: %f seconds\n", iter, end_time - start_time);
        }

        double** Anew = alloc_matrix(N, N);
        
        // Step 2: New A = R @ Q
        if (rank == 0) start_time = MPI_Wtime();
        parallel_matrix_multiplication(N, N, R_AtA, Q_AtA, Anew, comm);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("[Time] Iter %d (AtA): MatMul (R @ Q): %f seconds\n", iter, end_time - start_time);
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
        
    

        if (rank == 0) start_time = MPI_Wtime();
        parallel_matrix_multiplication(N, N, V, Q_AtA, Vtemp, comm);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("[Time] Iter %d (AtA): MatMul (Accumulate V): %f seconds\n", iter, end_time - start_time);
        }

        // Copy Vtemp into V
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                V[i][j] = Vtemp[i][j];
            }
        }
    

        free_matrix(Anew, N);
    }

    if(rank == 0){
        int mat_rank = min(N, M);
        printf("Eigenvalues:\n");
        for (size_t i = 0; i < M; i++){
            printf("\n");
            for (size_t j = 0; j < M; j++){
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
    free_matrix(eigvals, M);
}

int main(int argc, char* argv[]){

    int comm_sz; 
    int my_rank;
    double start_time, end_time, elapsed_time; 
    int num_matrices = 0;

    int R = 0, C = 0;
    // double** current_matrix = NULL;
    int elements = 0;

    FILE* dataset = NULL;
    FILE* results = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0){
        dataset = fopen("./svd-parallel/dataset.txt", "r");
        results = fopen("./svd-parallel/results_parallel.txt", "w");

        fprintf(results, "elements time\n");

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

        double** current_matrix = alloc_matrix(R, C);

        if(my_rank == 0){
            read_matrix(dataset, R, C, current_matrix);
        }

        double* flat_matrix = (double*)malloc(R * C * sizeof(double));

        if (my_rank == 0) {
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < C; j++) {
                    flat_matrix[i * C + j] = current_matrix[i][j];
                }
            }
        }

        MPI_Bcast(flat_matrix, R * C, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                current_matrix[i][j] = flat_matrix[i * C + j];
            }
        }

        free(flat_matrix); // read current matrix only by rank 0 and then bcast to every process once at the beginning, this avoid multiple bcast of columns inside qr decomposition


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
