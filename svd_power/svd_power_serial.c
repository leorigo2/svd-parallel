#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAX_ITER 10
#define EPS 1e-10

double norm2(double *x, int n);
void normalize(double *x, int n);
void compute_B(double *A, double *B, int rows, int cols, int method);
double power_iteration(double *B, double *v, int n);
void deflate(double *B, double *v, double lambda, int n);

int main(int argc, char **argv) {
    srand(1234);

    FILE *file = fopen("/datasets/dataset_same_r_c.txt", "r");
    if (!file) {
        printf("Error opening file 'dataset_same_r_c.txt'.\n");
        return 1;
    }

    int matrix_count;
    if (fscanf(file, "%d", &matrix_count) != 1) {
        printf("Error reading matrix count.\n");
        return 1;
    }
    
    printf("Processing %d matrices (Serial Version)...\n", matrix_count);

    double total_time = 0.0;

    for (int m_idx = 0; m_idx < matrix_count; m_idx++) {
        int M, N;
        if (fscanf(file, "%d %d", &M, &N) != 2) break;

        double *A = (double*)malloc(M * N * sizeof(double));
        for (int i = 0; i < M * N; i++) {
            fscanf(file, "%lf", &A[i]);
        }

        printf("\n=== Matrix %d/%d (%dx%d) ===\n", m_idx + 1, matrix_count, M, N);

        clock_t start = clock();

        int method = (N >= M) ? 1 : 0; 
        int dim = (method == 1) ? M : N; // min(M, N)

        double *B = (double*)malloc(dim * dim * sizeof(double));
        
        // Compute B = A*A^T or A^T*A
        compute_B(A, B, M, N, method);

        int K = dim;
        
        for (int k = 0; k < K; k++) {
            double *v = (double*)malloc(dim * sizeof(double));
            // Initialize random vector
            for(int i=0; i<dim; i++) v[i] = (double)rand() / RAND_MAX;

            double lambda = power_iteration(B, v, dim);
            double sigma = sqrt(fabs(lambda)); // lambda is eigenvalue of B (sigma^2)

            if (sigma < EPS) {
                free(v);
                break;
            }

            // printf("sigma_%d = %f\n", k+1, sigma);

            deflate(B, v, lambda, dim);
            free(v);
        }

        free(B);
        free(A);

        clock_t end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Execution time: %f s\n", time_taken);
        total_time += time_taken;
    }

    printf("Total execution time: %f s\n", total_time);
    fclose(file);
    return 0;
}

double norm2(double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrt(sum);
}

void normalize(double *x, int n) {
    double norm = norm2(x, n);
    if (norm > 1e-15) {
        for (int i = 0; i < n; i++) x[i] /= norm;
    }
}

void compute_B(double *A, double *B, int rows, int cols, int method) {
    // method 0: B = A^T * A (size cols x cols)
    // method 1: B = A * A^T (size rows x rows)
    
    int dim = (method == 1) ? rows : cols;

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            double sum = 0.0;
            if (method == 0) {
                // B[i][j] = dot(Col i of A, Col j of A)
                // Col i elements are A[k*cols + i]
                for (int k = 0; k < rows; k++) {
                    sum += A[k * cols + i] * A[k * cols + j];
                }
            } else {
                // B[i][j] = dot(Row i of A, Row j of A)
                // Row i elements are A[i*cols + k]
                for (int k = 0; k < cols; k++) {
                    sum += A[i * cols + k] * A[j * cols + k];
                }
            }
            B[i * dim + j] = sum;
        }
    }
}

double power_iteration(double *B, double *v, int n) {
    double *y = (double*)malloc(n * sizeof(double));
    double lambda = 0.0;
    double lambda_old = 0.0;

    normalize(v, n);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // y = B * v
        for (int i = 0; i < n; i++) {
            y[i] = 0.0;
            for (int j = 0; j < n; j++) {
                y[i] += B[i * n + j] * v[j];
            }
        }

        lambda = norm2(y, n);
        
        if (lambda < 1e-15) break;

        // v = y / lambda
        for (int i = 0; i < n; i++) {
            v[i] = y[i] / lambda;
        }

        if (fabs(lambda - lambda_old) < 1e-8) break;
        lambda_old = lambda;
    }

    free(y);
    return lambda;
}

void deflate(double *B, double *v, double lambda, int n) {
    // B = B - lambda * v * v^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i * n + j] -= lambda * v[i] * v[j];
        }
    }
}
