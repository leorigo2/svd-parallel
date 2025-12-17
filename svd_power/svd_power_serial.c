#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ROWS 4
#define COLS 3
#define METHOD COLS>=ROWS
#define MAX_DIM (COLS>ROWS) ? COLS : ROWS
#define DIM_X ((COLS>ROWS) ? ROWS: COLS)
#define MAX_ITER 200

double singolar_value(double lambda);
void transpose(double matrix[ROWS][COLS], double transposed[COLS][ROWS]);
void computeB(double matrix[ROWS][COLS], double B[DIM_X][DIM_X]);
void printMatrix(double matrix[MAX_DIM][MAX_DIM], int rows, int cols);
void init_x(double * x);
void matrix_array_prod(double B[DIM_X][DIM_X], double * random_vector, double * result);
double vector_norm(double * result);
void normalize_vector(double * result);
double power_iteration(double B[DIM_X][DIM_X], double * random_vector);
void copy_array(double * src, double * dst);
void mult_A_vector(double A[ROWS][COLS], double *vec, double *result);
void mult_At_vector(double At[COLS][ROWS], double *vec, double *result);
void compute_singular_vectors(double A[ROWS][COLS], double * eigen_vector, double sigma, double *u, double *v);
void deflate_matrix(double B[DIM_X][DIM_X], double lambda, double *u);

int main (int argc, char ** argv){
    srand(time(NULL));

    double matrix[ROWS][COLS] = {
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12
        };

    printf("matrix:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }

    double B[DIM_X][DIM_X];
    computeB(matrix, B);
    
    double sigmas[DIM_X];
    double U[DIM_X][ROWS];
    double V[DIM_X][COLS];
    
    for (int k = 0; k < DIM_X; k++) {
        double random_vector[DIM_X];
        init_x(random_vector);
        
        // Power iteration
        double lambda = power_iteration(B, random_vector);
        double sigma = singolar_value(lambda);
        sigmas[k] = sigma;
        
        double u[MAX_DIM];
        double v[MAX_DIM];
        compute_singular_vectors(matrix, random_vector, sigma, u, v);
        
        for (int i = 0; i < ROWS; i++) U[k][i] = u[i];
        for (int i = 0; i < COLS; i++) V[k][i] = v[i];
        
        printf("\nSingular value %d = %f\n", k+1, sigma);
        printf("Left singular vector u%d = ", k+1);
        for (int i = 0; i < ROWS; i++) printf("%f ", u[i]);
        printf("\nRight singular vector v%d = ", k+1);
        for (int i = 0; i < COLS; i++) printf("%f ", v[i]);
        printf("\n");
        
        deflate_matrix(B, lambda, random_vector);
    }
    
    printf("\n=== SVD Summary ===\n");
    printf("Singular values: ");
    for (int i = 0; i < DIM_X; i++) printf("%f ", sigmas[i]);
    printf("\n");
    
    return 0;
}

// Function to transpose a matrix
void transpose(double src[ROWS][COLS], double dest[COLS][ROWS]) {
    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++) {
            dest[i][j] = src[j][i];
        }
    }
}

// Function to compute B
void computeB(double A[ROWS][COLS], double B[DIM_X][DIM_X]) {
    double At[COLS][ROWS];
    transpose(A, At);

    if (METHOD == 0) {  // Caso: ROWS > COLS → B = A^T * A
        for (int i = 0; i < DIM_X; i++) {
            for (int j = 0; j < DIM_X; j++) {
                B[i][j] = 0.0;
                for (int k = 0; k < ROWS; k++) {
                    B[i][j] += At[i][k] * A[k][j];
                }
            }
        }
        printf("B = A^T * A (COLS x COLS)\n");
        for (int i = 0; i < DIM_X; i++) {
            for (int j = 0; j < DIM_X; j++) {
                printf("%f ", B[i][j]);
            }
            printf("\n");
        }
    } 
    else {  // Caso: ROWS <= COLS → B = A * A^T
        for (int i = 0; i < DIM_X; i++) {
            for (int j = 0; j < DIM_X; j++) {
                B[i][j] = 0.0;
                for (int k = 0; k < COLS; k++) {
                    B[i][j] += A[i][k] * At[k][j];
                }
            }
        }
        printf("B = A * A^T (ROWS x ROWS)\n");
        for (int i = 0; i < DIM_X; i++) {
            for (int j = 0; j < DIM_X; j++) {
                printf("%f ", B[i][j]);
            }
            printf("\n");
        }
    }
}

// Function to compute matrix * array
void matrix_array_prod(double B[DIM_X][DIM_X], double * random_vector, double * result){
    for(int i=0; i<DIM_X; i++){
        result[i] = 0.0;
        for(int j=0; j<DIM_X; j++){
            result[i]+= random_vector[j]*B[i][j];
        }
    }

}

// Norm L2
double vector_norm(double * result){
    double sum=0.0;
    for(int i=0; i<DIM_X; i++)sum+=result[i]*result[i];
    return sqrt(sum);
}

// Normalization with L2
void normalize_vector(double * result){
    double norm = vector_norm(result);
    if(norm==0){
        printf("Erorre! Norm=0");
        return;
    }
    for(int i=0; i<DIM_X; i++){
        result[i] /= norm;
    }
}

// Iterations to find eigen value
double power_iteration(double B[DIM_X][DIM_X], double * random_vector){
    double result[DIM_X];
    double lambda = 0.0;
    for(int i=0; i<MAX_ITER; i++){
        matrix_array_prod(B, random_vector, result);
        lambda = vector_norm(result);
        normalize_vector(result);
        copy_array(result, random_vector);
    }

    return lambda;
}

// function to compute the other eigen vector
void compute_singular_vectors(double A[ROWS][COLS], double * eigen_vector, double sigma, double *u, double *v){
     if (METHOD == 0) {  // B = A^T·A → eigen_vector è v
        for (int i = 0; i < COLS; i++) {
            v[i] = eigen_vector[i];
        }
        
        // Calcola u = (A · v) / σ
        mult_A_vector(A, v, u);
        for (int i = 0; i < ROWS; i++) {
            u[i] = u[i] / sigma;
        }
    }
    else {  // B = A·A^T → eigen_vector è u
        for (int i = 0; i < ROWS; i++) {
            u[i] = eigen_vector[i];
        }
        
        // Calcola v = (A^T · u) / σ
        double At[COLS][ROWS];
        transpose(A, At);
        mult_At_vector(At, u, v);
        for (int i = 0; i < COLS; i++) {
            v[i] = v[i] / sigma;
        }
    }
}

void mult_At_vector(double At[COLS][ROWS], double *vec, double *result) {
    for (int i = 0; i < COLS; i++) {
        result[i] = 0.0;
        for (int j = 0; j < ROWS; j++) {
            result[i] += At[i][j] * vec[j];
        }
    }
}

void mult_A_vector(double A[ROWS][COLS], double *vec, double *result) {
    for (int i = 0; i < ROWS; i++) {
        result[i] = 0.0;
        for (int j = 0; j < COLS; j++) {
            result[i] += A[i][j] * vec[j];
        }
    }
}

void copy_array(double * src, double * dst){
    for(int i=0; i<DIM_X; i++){
        dst[i] = src[i];
    }
}

double singolar_value(double lambda){
    return sqrt(lambda);
}

void init_x(double * x){
    for(int i=0; i<DIM_X; i++) x[i] = 2.0* rand()/ RAND_MAX -1.0;
}

// Function to remove the last value found
void deflate_matrix(double B[DIM_X][DIM_X], double lambda, double *u) {
    // B = B - λ · (u·u^T)
    for (int i = 0; i < DIM_X; i++) {
        for (int j = 0; j < DIM_X; j++) {
            B[i][j] = B[i][j] - lambda * u[i] * u[j];
        }
    }
}
