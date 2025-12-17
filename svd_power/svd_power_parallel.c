#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define MAX_IT 200
#define EPS 1e-10

void power_svd(double *A_local, int M, int N, int local_rows, MPI_Comm comm);
double norm2(double *x, int N);
double power_iteration(double *A_local, double *v, int local_rows, int N, MPI_Comm comm);
void compute_u(double *A_local, double *v, double *u_local, int local_rows, int N, double sigma);
void deflate(double *A_local, double *u_local, double *v, double sigma, int local_rows, int N);

int main(int argc, char *argv[]) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(1234);  // Same seed for all processes for the random initialization

    int M, N;
    double *A = NULL;

    // Input is read only by rank 0
    if (rank == 0) {
        M = 4;
        N = 3;

        A = malloc(M * N * sizeof(double));
        double data[12] = {
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12
        };
        for (int i = 0; i < M * N; i++)
            A[i] = data[i];

        printf("MPI processes: %d\n", size);
        printf("Matrix size: M=%d, N=%d\n", M, N);
        
        printf("Matrix A:\n");
        for(int i=0; i<M; i++) {
            for(int j=0; j<N; j++) {
                printf("%6.2f ", A[i*N + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Sending sizes to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Dividing the matrix A among processes
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int base = M / size;
    int rem  = M % size;

    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = base + (i < rem ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = offset;
        offset += rows * N;
    }

    int local_rows = sendcounts[rank] / N;
    double *A_local = malloc(local_rows * N * sizeof(double));

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, A_local, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
        free(A);

    printf("[Rank %d] local_rows = %d\n", rank, local_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    power_svd(A_local, M, N, local_rows, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("\nTotal execution time: %f s\n", t1 - t0);

    free(A_local);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}

double norm2(double *x, int N) {
    double sum = 0.0;

    for (int i = 0; i < N; i++)
        sum += x[i] * x[i];

    return sqrt(sum);
}

double power_iteration(double *A_local, double *v, int local_rows, int N, MPI_Comm comm) {
    double *y = malloc(N * sizeof(double));
    double *w_local = malloc(local_rows * sizeof(double));
    double *z_local = malloc(N * sizeof(double));
    double lambda = 0.0, lambda_old = 0.0;

    for (int it = 0; it < MAX_IT; it++) {
        // w_local = A_local * v
        for (int i = 0; i < local_rows; i++) {
            w_local[i] = 0.0;
            for (int j = 0; j < N; j++) {
                w_local[i] += A_local[i*N + j] * v[j];
            }
        }

        // z_local = A_local^T * w_local
        for (int j = 0; j < N; j++) {
            z_local[j] = 0.0;
            for (int i = 0; i < local_rows; i++) {
                z_local[j] += A_local[i*N + j] * w_local[i];
            }
        }

        // y = Allreduce(z_local)
        MPI_Allreduce(z_local, y, N, MPI_DOUBLE, MPI_SUM, comm);

        lambda = norm2(y, N);

        if (lambda < EPS)
            break;

        for (int i = 0; i < N; i++)
            v[i] = y[i] / lambda;

        if (fabs(lambda - lambda_old) < 1e-8)
            break;

        lambda_old = lambda;
    }

    free(w_local);
    free(z_local);
    free(y);
    return lambda;
}

void compute_u(double *A_local, double *v, double *u_local,
            int local_rows, int N, double sigma) {

    for (int i = 0; i < local_rows; i++) {
        u_local[i] = 0.0;
        for (int j = 0; j < N; j++)
            u_local[i] += A_local[i*N + j] * v[j];
        u_local[i] /= sigma;
    }
}

void deflate(double *A_local, double *u_local, double *v, double sigma, int local_rows, int N) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            A_local[i*N + j] -= sigma * u_local[i] * v[j];
        }
    }
}

void power_svd(double *A_local, int M, int N, int local_rows, MPI_Comm comm) {

    int rank;
    MPI_Comm_rank(comm, &rank);

    int K = (M < N) ? M : N;

    if (rank == 0) {

        printf("Computing %d singular values\n", K);
    }

    for (int k = 0; k < K; k++) {
        double *v = malloc(N * sizeof(double));
        for (int i = 0; i < N; i++)
            v[i] = rand()/(double)RAND_MAX;  // random vector initialization

        double lambda = power_iteration(A_local, v, local_rows, N, comm);
        double sigma = sqrt(lambda);

        double *u_local = malloc(local_rows * sizeof(double));
        compute_u(A_local, v, u_local, local_rows, N, sigma);

        if (rank == 0)
            printf("sigma_%d = %.8e\n", k+1, sigma);

        deflate(A_local, u_local, v, sigma, local_rows, N);

        free(v);
        free(u_local);
    }
}
