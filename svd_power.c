#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define ROWS 2
#define COLS 2

void transpose(int matrix[ROWS][COLS], int transposed[COLS][ROWS]);
void computeB(int matrix[ROWS][COLS], int B[ROWS][COLS]);
void printMatrix(int matrix[ROWS][COLS], int rows, int cols);

int main (int argc, char ** argv){
    int matrix[ROWS][COLS] = { 1, 2,
                                3, 4 };

    printf("matrix:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

    int B[ROWS][COLS];
    computeB(matrix, B);
    
    return 0;
}

void transpose(int src[ROWS][COLS], int dest[COLS][ROWS]) {
    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++) {
            dest[i][j] = src[j][i];
        }
    }
}

void computeB(int matrix[ROWS][COLS], int B[ROWS][COLS]){
    int transposed[COLS][ROWS];
    transpose(matrix, transposed);
    printf("transposed:\n");
    printMatrix(transposed, ROWS, COLS);
    for(int i=0; i<ROWS; i++){
        for(int j=0; j<COLS; j++){
            B[i][j] = 0;
            for(int k=0; k<COLS; k++){
                B[i][j] += matrix[i][k] * transposed[k][j];
            }
        }
    }
    printf("B (matrix * transposed):\n");
    printMatrix(B, ROWS, COLS);
}

void printMatrix(int matrix[ROWS][COLS], int rows, int cols){
    for(int i=0 ; i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}