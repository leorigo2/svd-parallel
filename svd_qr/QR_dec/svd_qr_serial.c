#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

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


void QR_Decomposition(size_t n, double **A, double **Q, double **R) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            R[i][j] = 0.0;

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
        
        norm = sqrt(norm);
        R[i][i] = norm;

        // Normalize Q[:, i]
        for (size_t k = 0; k < n; k++){
            if(norm == 0) continue; 
            Q[k][i] = u[k] / norm;
        }
    }
}


void QR_SVD(double** A, int M, int N){

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
        double **Anew = alloc_matrix(M, M);

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
        free_matrix(Anew, M);
    }
    for (size_t i = 0; i < M; i++)
        eigvals[i][i] = AAt[i][i];


    // Compute AtA eigenvector
    for(int iter = 0; iter < iterations; iter++){
        // Step 1: QR decomposition
        QR_Decomposition(N, AtA, Q_AtA, R_AtA);

        // Step 2: New A = R @ Q
        double **Anew = alloc_matrix(N, N);
                
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
        free_matrix(Anew, N);
    }
    int rank = min(N, M);
    printf("Eigenvalues:");
    for (size_t i = 0; i < M; i++){
        printf("\n");
        for (size_t j = 0; j < M; j++){
            if(i == j) printf("%f   ", sqrt(eigvals[i][j]));
        }
    }
    /*
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
            printf("%f  ", V[i][j]);
        }
    }
    */
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

int main()
{
    int R=0, C=0;
    int num_matrices=0;
    int elements=0;

    FILE* results = NULL;
    FILE* dataset = NULL;

    dataset = fopen("dataset.txt", "r");
    results = fopen("results_serial.txt", "w");

    fprintf(results, "elements time\n");

    fscanf(dataset, "%d", &num_matrices);

    for(int k=0; k<num_matrices; k++){
        fscanf(dataset, "%d %d", &R, &C);
        elements = R*C;
        double** A = alloc_matrix(R, C);

        read_matrix(dataset, R, C, A);

        clock_t start = clock();

        QR_SVD(A, R, C); 

        clock_t end = clock();

        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        fprintf(results, "%d %f\n", elements, elapsed);

        free_matrix(A, R);
    }

    fclose(results);
    fclose(dataset);

    return 0;
}

