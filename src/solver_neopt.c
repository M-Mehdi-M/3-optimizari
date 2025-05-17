/*
 * Tema 2 ASC
 * 2025 Spring
 */
#include "utils.h"

/*
 * Unoptimized implementation that follows the mathematical formulas.
 * This code is slow but easy to understand.
 */
double* my_solver(int N, double *A, double *B, double *x) {
    // Allocate memory for all matrices and vectors used in computation
    double *At = malloc(N * N * sizeof(double));  // For storing A transposed
    double *C = malloc(N * N * sizeof(double));   // For B * At
    double *Ct = malloc(N * N * sizeof(double));  // For storing C transposed
    double *D = malloc(N * N * sizeof(double));   // For Ct * A
    double *y = malloc(N * sizeof(double));       // Temporary vector for iterative step
    double *tmp = malloc(N * sizeof(double));     // Another temporary vector for iterations
    double *out = malloc(N * sizeof(double));     // Final output vector
    int i, j, k;

    // Compute the transpose of matrix A and store in At
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            At[i*N + j] = A[j*N + i];

    // Compute matrix C = B * At
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++) {
            C[i*N + j] = 0;
            for(k = 0; k < N; k++)
                C[i*N + j] += B[i*N + k] * At[k*N + j];
        }

    // Compute the transpose of matrix C and store in Ct
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            Ct[i*N + j] = C[j*N + i];

    // Compute matrix D = Ct * A
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++) {
            D[i*N + j] = 0;
            for(k = 0; k < N; k++)
                D[i*N + j] += Ct[i*N + k] * A[k*N + j];
        }

    // Copy input vector x into tmp for use in the iterative step
    for(i = 0; i < N; i++)
        tmp[i] = x[i];

    // Perform N iterations:
    //   y = Ct * x
    //   x = C * y
    // This repeatedly multiplies the vector by the matrices
    for(i = 0; i < N; i++) {
        // Compute y = Ct * x
        for(j = 0; j < N; j++) {
            y[j] = 0;
            for(k = 0; k < N; k++)
                y[j] += Ct[j*N + k] * tmp[k];
        }
        // Compute x = C * y
        for(j = 0; j < N; j++) {
            tmp[j] = 0;
            for(k = 0; k < N; k++)
                tmp[j] += C[j*N + k] * y[k];
        }
    }

    // Compute final output: out = D * x
    for(i = 0; i < N; i++) {
        out[i] = 0;
        for(j = 0; j < N; j++)
            out[i] += D[i*N + j] * tmp[j];
    }

    free(At);
    free(C);
    free(Ct);
    free(D);
    free(y);
    free(tmp);

    return out;
}

