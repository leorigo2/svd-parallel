#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ROWS 2
#define COLS 3
#define METHOD COLS>=ROWS
#define MAX_DIM (COLS>ROWS) ? COLS : ROWS
#define DIM_X ((COLS>ROWS) ? ROWS: COLS)

int min(int a, int b);
void transpose(double matrix[ROWS][COLS], double transposed[COLS][ROWS]);
void computeB(double matrix[ROWS][COLS], double B[DIM_X][DIM_X]);
void printMatrix(double matrix[MAX_DIM][MAX_DIM], int rows, int cols);
void init_x(double * x);
void matrix_x_prod(double B[DIM_X][DIM_X], double * random_vector, double * result);

int main (int argc, char ** argv){
    srand(time(NULL));
    
    printf("%d\n", DIM_X);

    printf("%d\n", METHOD);
    double random_vector[DIM_X];
    init_x(random_vector);
    for(int i=0; i<DIM_X; i++){
        printf("%f ", random_vector[i]);
    }
    printf("\n");

    double matrix[ROWS][COLS] = { 1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0 };

    printf("matrix:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }

    double B[DIM_X][DIM_X];
    computeB(matrix, B);
    
    double result[DIM_X];
    matrix_x_prod(B, random_vector, result);

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

void matrix_x_prod(double B[DIM_X][DIM_X], double * random_vector, double * result){
    for(int i=0; i<DIM_X; i++){
        result[i] = 0.0;
        for(int j=0; j<DIM_X; j++){
            result[i] = result[i] + random_vector[j]*B[i][j];
        }
    }

    for(int i=0; i<DIM_X; i++){
        printf("%f ", result[i]);
    }
    printf("\n");
}

void init_x(double * x){
    for(int i=0; i<DIM_X; i++) x[i] = 2.0* rand()/ RAND_MAX -1.0;
}

int min(int a, int b){
    return (a<b) ? a : b;
}