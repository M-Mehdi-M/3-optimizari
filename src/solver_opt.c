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
    double* restrict C  = calloc(N * N, sizeof(double));   // B * A^T
    double* restrict Ct = malloc(N * N * sizeof(double));  // Transposed C
    double* restrict D  = calloc(N * N, sizeof(double));   // Ct * A
    double* restrict y1 = malloc(N * sizeof(double));      // Work buffer 1
    double* restrict y2 = malloc(N * sizeof(double));      // Work buffer 2
    double* restrict out = malloc(N * sizeof(double));     // Final result
    int i, j, k;

    // Compute the transpose of A (At = A^T), using 2x2 blocks for better speed
    for (i = 0; i < N; i += 2) {
        for (j = 0; j < N; j += 2) {
            At[j*N + i]       = A[i*N + j];
            At[j*N + i+1]     = A[(i+1)*N + j];
            At[(j+1)*N + i]   = A[i*N + j+1];
            At[(j+1)*N + i+1] = A[(i+1)*N + j+1];
        }
    }

    // Compute C = B * At, with inner loop unrolled by 16 for speed
    for (i = 0; i < N; i++) {
        for (k = 0; k < N; k++) {
            double b = B[i*N + k];
            double* restrict C_row = C + i*N;
            double* restrict At_row = At + k*N;
            int j = 0;
            for (; j + 15 < N; j += 16) {
                C_row[j]    += b * At_row[j];
                C_row[j+1]  += b * At_row[j+1];
                C_row[j+2]  += b * At_row[j+2];
                C_row[j+3]  += b * At_row[j+3];
                C_row[j+4]  += b * At_row[j+4];
                C_row[j+5]  += b * At_row[j+5];
                C_row[j+6]  += b * At_row[j+6];
                C_row[j+7]  += b * At_row[j+7];
                C_row[j+8]  += b * At_row[j+8];
                C_row[j+9]  += b * At_row[j+9];
                C_row[j+10] += b * At_row[j+10];
                C_row[j+11] += b * At_row[j+11];
                C_row[j+12] += b * At_row[j+12];
                C_row[j+13] += b * At_row[j+13];
                C_row[j+14] += b * At_row[j+14];
                C_row[j+15] += b * At_row[j+15];
            }
            // Handle leftover elements
            for (; j < N; j++) {
                C_row[j] += b * At_row[j];
            }
        }
    }

    // Compute transpose of C (Ct = C^T) using 2x2 blocks
    for (i = 0; i < N; i += 2) {
        for (j = 0; j < N; j += 2) {
            Ct[j*N + i]       = C[i*N + j];
            Ct[j*N + i+1]     = C[(i+1)*N + j]; 
            Ct[(j+1)*N + i]   = C[i*N + j+1];
            Ct[(j+1)*N + i+1] = C[(i+1)*N + j+1];
        }
    }

    // Compute D = Ct * A, again with unrolled inner loop
    for (i = 0; i < N; i++) {
        for (k = 0; k < N; k++) {
            double ct = Ct[i*N + k];
            double* restrict D_row = D + i*N;
            double* restrict A_row = A + k*N;
            int j = 0;
            for (; j + 15 < N; j += 16) {
                D_row[j]    += ct * A_row[j];
                D_row[j+1]  += ct * A_row[j+1];
                D_row[j+2]  += ct * A_row[j+2];
                D_row[j+3]  += ct * A_row[j+3];
                D_row[j+4]  += ct * A_row[j+4];
                D_row[j+5]  += ct * A_row[j+5];
                D_row[j+6]  += ct * A_row[j+6];
                D_row[j+7]  += ct * A_row[j+7];
                D_row[j+8]  += ct * A_row[j+8];
                D_row[j+9]  += ct * A_row[j+9];
                D_row[j+10] += ct * A_row[j+10];
                D_row[j+11] += ct * A_row[j+11];
                D_row[j+12] += ct * A_row[j+12];
                D_row[j+13] += ct * A_row[j+13];
                D_row[j+14] += ct * A_row[j+14];
                D_row[j+15] += ct * A_row[j+15];
            }
            for (; j < N; j++) {
                D_row[j] += ct * A_row[j];
            }
        }
    }

    // Copy input vector x to y1 (first work buffer)
    for (i = 0; i < N; i++) {
        y1[i] = x[i];
    }

    // Run N steps of: y = Ct*x, x = C*y (alternating buffers for speed)
    double* restrict curr = y1;
    double* restrict next = y2;
    for (i = 0; i < N; i++) {
        // Compute next = Ct * curr (matrix-vector multiplication)
        for (j = 0; j < N; j++) {
            double* restrict Ct_row = Ct + j*N;
            double sum = 0.0;
            int k = 0;
            for (; k + 15 < N; k += 16) {
                sum += Ct_row[k]    * curr[k] +
                       Ct_row[k+1]  * curr[k+1] +
                       Ct_row[k+2]  * curr[k+2] +
                       Ct_row[k+3]  * curr[k+3] +
                       Ct_row[k+4]  * curr[k+4] +
                       Ct_row[k+5]  * curr[k+5] +
                       Ct_row[k+6]  * curr[k+6] +
                       Ct_row[k+7]  * curr[k+7] +
                       Ct_row[k+8]  * curr[k+8] +
                       Ct_row[k+9]  * curr[k+9] +
                       Ct_row[k+10] * curr[k+10] +
                       Ct_row[k+11] * curr[k+11] +
                       Ct_row[k+12] * curr[k+12] +
                       Ct_row[k+13] * curr[k+13] +
                       Ct_row[k+14] * curr[k+14] +
                       Ct_row[k+15] * curr[k+15];
            }
            for (; k < N; k++) {
                sum += Ct_row[k] * curr[k];
            }
            next[j] = sum;
        }
        // Swap the two buffers for next iteration (no copy needed)
        double* temp = curr;
        curr = next;
        next = temp;
    }

    // Final matrix-vector: out = D * curr
    for (i = 0; i < N; i++) {
        double* restrict D_row = D + i*N;
        double sum = 0.0;
        int j = 0;
        for (; j + 15 < N; j += 16) {
            sum += D_row[j]    * curr[j] +
                   D_row[j+1]  * curr[j+1] +
                   D_row[j+2]  * curr[j+2] +
                   D_row[j+3]  * curr[j+3] +
                   D_row[j+4]  * curr[j+4] +
                   D_row[j+5]  * curr[j+5] +
                   D_row[j+6]  * curr[j+6] +
                   D_row[j+7]  * curr[j+7] +
                   D_row[j+8]  * curr[j+8] +
                   D_row[j+9]  * curr[j+9] +
                   D_row[j+10] * curr[j+10] +
                   D_row[j+11] * curr[j+11] +
                   D_row[j+12] * curr[j+12] +
                   D_row[j+13] * curr[j+13] +
                   D_row[j+14] * curr[j+14] +
                   D_row[j+15] * curr[j+15];
        }
        for (; j < N; j++) {
            sum += D_row[j] * curr[j];
        }
        out[i] = sum;
    }

    free(At);
    free(C);
    free(Ct);
    free(D);
    free(y1);
    free(y2);

    return out;
}

