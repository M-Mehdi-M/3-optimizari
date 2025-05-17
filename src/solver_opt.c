/*
 * Tema 2 ASC
 * 2025 Spring
 */
#include <stdlib.h>
#include "utils.h"

/*
 * Optimized implementation using loop unrolling and cache-friendly memory layout.
 * Computes: out = D * x', where D = (B*A^T)^T * A, x' is x after N iterative matrix-vector steps.
 */
double* my_solver(int N, double* restrict A, double* restrict B, double* restrict x) {
    // Allocate memory for all needed matrices and vectors
    double* restrict At = malloc(N * N * sizeof(double));  // Transposed A
    double* restrict C  = malloc(N * N * sizeof(double));  // B * A^T
    double* restrict Ct = malloc(N * N * sizeof(double));  // Transposed C
    double* restrict D  = malloc(N * N * sizeof(double));  // C^t * A
    double* restrict y = malloc(N * sizeof(double));       // Work buffer
    double* restrict tmp = malloc(N * sizeof(double));     // Current vector
    double* restrict out = malloc(N * sizeof(double));     // Final result
    int i, j, k;

    // Initialize C and D matrices to zero
    for (i = 0; i < N * N; i++) {
        C[i] = 0.0;
        D[i] = 0.0;
    }

    // Compute the transpose of A (At = A^T)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            At[i*N + j] = A[j*N + i];
        }
    }

    // Compute C = B * At with unrolled inner loop
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k += 8) {
                if (k + 7 < N) {
                    sum += B[i*N + k] * At[k*N + j] +
                           B[i*N + k+1] * At[(k+1)*N + j] +
                           B[i*N + k+2] * At[(k+2)*N + j] +
                           B[i*N + k+3] * At[(k+3)*N + j] +
                           B[i*N + k+4] * At[(k+4)*N + j] +
                           B[i*N + k+5] * At[(k+5)*N + j] +
                           B[i*N + k+6] * At[(k+6)*N + j] +
                           B[i*N + k+7] * At[(k+7)*N + j];
                } else {
                    // Handle remaining elements
                    for (int m = k; m < N; m++) {
                        sum += B[i*N + m] * At[m*N + j];
                    }
                    break;
                }
            }
            C[i*N + j] = sum;
        }
    }

    // Compute transpose of C (Ct = C^T)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            Ct[i*N + j] = C[j*N + i];
        }
    }

    // Compute D = Ct * A
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k += 8) {
                if (k + 7 < N) {
                    sum += Ct[i*N + k] * A[k*N + j] +
                           Ct[i*N + k+1] * A[(k+1)*N + j] +
                           Ct[i*N + k+2] * A[(k+2)*N + j] +
                           Ct[i*N + k+3] * A[(k+3)*N + j] +
                           Ct[i*N + k+4] * A[(k+4)*N + j] +
                           Ct[i*N + k+5] * A[(k+5)*N + j] +
                           Ct[i*N + k+6] * A[(k+6)*N + j] +
                           Ct[i*N + k+7] * A[(k+7)*N + j];
                } else {
                    // Handle remaining elements
                    for (int m = k; m < N; m++) {
                        sum += Ct[i*N + m] * A[m*N + j];
                    }
                    break;
                }
            }
            D[i*N + j] = sum;
        }
    }

    // Copy input vector x to tmp
    for (i = 0; i < N; i++) {
        tmp[i] = x[i];
    }

    // Run N steps of: y = Ct*x, x = C*y
    for (i = 0; i < N; i++) {
        // y = Ct * tmp
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k += 8) {
                if (k + 7 < N) {
                    sum += Ct[j*N + k] * tmp[k] +
                           Ct[j*N + k+1] * tmp[k+1] +
                           Ct[j*N + k+2] * tmp[k+2] +
                           Ct[j*N + k+3] * tmp[k+3] +
                           Ct[j*N + k+4] * tmp[k+4] +
                           Ct[j*N + k+5] * tmp[k+5] +
                           Ct[j*N + k+6] * tmp[k+6] +
                           Ct[j*N + k+7] * tmp[k+7];
                } else {
                    // Handle remaining elements
                    for (int m = k; m < N; m++) {
                        sum += Ct[j*N + m] * tmp[m];
                    }
                    break;
                }
            }
            y[j] = sum;
        }

        // tmp = C * y
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k += 8) {
                if (k + 7 < N) {
                    sum += C[j*N + k] * y[k] +
                           C[j*N + k+1] * y[k+1] +
                           C[j*N + k+2] * y[k+2] +
                           C[j*N + k+3] * y[k+3] +
                           C[j*N + k+4] * y[k+4] +
                           C[j*N + k+5] * y[k+5] +
                           C[j*N + k+6] * y[k+6] +
                           C[j*N + k+7] * y[k+7];
                } else {
                    // Handle remaining elements
                    for (int m = k; m < N; m++) {
                        sum += C[j*N + m] * y[m];
                    }
                    break;
                }
            }
            tmp[j] = sum;
        }
    }

    // Final matrix-vector: out = D * tmp
    for (i = 0; i < N; i++) {
        double sum = 0.0;
        for (j = 0; j < N; j += 8) {
            if (j + 7 < N) {
                sum += D[i*N + j] * tmp[j] +
                       D[i*N + j+1] * tmp[j+1] +
                       D[i*N + j+2] * tmp[j+2] +
                       D[i*N + j+3] * tmp[j+3] +
                       D[i*N + j+4] * tmp[j+4] +
                       D[i*N + j+5] * tmp[j+5] +
                       D[i*N + j+6] * tmp[j+6] +
                       D[i*N + j+7] * tmp[j+7];
            } else {
                // Handle remaining elements
                for (int m = j; m < N; m++) {
                    sum += D[i*N + m] * tmp[m];
                }
                break;
            }
        }
        out[i] = sum;
    }

    free(At);
    free(C);
    free(Ct);
    free(D);
    free(y);
    free(tmp);

    return out;
}
