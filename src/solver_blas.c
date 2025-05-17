/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"
#include <cblas.h>

/*
 * BLAS implementation using BLAS library.
 * Uses cblas_dgemm for matrix-matrix multiplication and cblas_dgemv for matrix-vector multiplication.
 */
double* my_solver(int N, double *A, double *B, double *x) {
    // Allocate memory for all matrices and vectors
    double *At = malloc(N * N * sizeof(double));  // Transposed A
    double *C = malloc(N * N * sizeof(double));   // B * At
    double *Ct = malloc(N * N * sizeof(double));  // Transposed C
    double *D = malloc(N * N * sizeof(double));   // Ct * A
    double *y = malloc(N * sizeof(double));       // Work vector
    double *tmp = malloc(N * sizeof(double));     // Work vector
    double *out = malloc(N * sizeof(double));     // Output vector
    int i;

    // Compute At = transpose(A)
    for (i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            At[i*N + j] = A[j*N + i];

    // Compute C = B * At using BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, B, N, At, N, 0.0, C, N);

    // Compute Ct = transpose(C)
    for (i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Ct[i*N + j] = C[j*N + i];

    // Compute D = Ct * A using BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, Ct, N, A, N, 0.0, D, N);

    // Copy input vector x into tmp
    for (i = 0; i < N; i++)
        tmp[i] = x[i];

    // Perform N iterations:
    //   y = Ct * x
    //   x = C * y
    for (i = 0; i < N; i++) {
        // y = Ct * x
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, Ct, N, tmp, 1, 0.0, y, 1);
        // x = C * y
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, C, N, y, 1, 0.0, tmp, 1);
    }

    // Compute final result: out = D * x
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, D, N, tmp, 1, 0.0, out, 1);

    free(At);
    free(C);
    free(Ct);
    free(D);
    free(y);
    free(tmp);

    return out;
}

