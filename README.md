# 3-optimizari

I implemented and compared three versions of the same matrix computation, to see how much difference low-level optimization and using a proper BLAS library make.

The computation itself: given a matrix A, a matrix B, and a vector x, it computes C = B · Aᵀ, D = Cᵀ · A, then iterates x through C and Cᵀ N times, and finally multiplies by D to get the result.

**General approach:**
solver_neopt.c — the naive, straight-from-the-math implementation, with plain triple-nested loops.
solver_opt.c — a hand-optimized version using restrict pointers and 8-way manual loop unrolling on every matrix multiplication.
solver_blas.c — uses the ATLAS BLAS library (cblas_dgemm and cblas_dgemv) instead of hand-written loops.
main.c and compare.c are the provided test harness: they generate random test matrices from a seed, time each solver, and check that all three produce the same result within a tolerance.

I consider this assignment useful for understanding: how compiler-level optimizations like loop unrolling and pointer aliasing actually affect performance, how cache behavior changes between naive and optimized code, and how much a properly optimized library (BLAS) can outperform even carefully hand-tuned loops.

**Results:** on the same input (N=400), the naive version took about 42 seconds, the hand-optimized version about 25 seconds, and the BLAS version about 3.5 seconds — confirmed with Valgrind's Cachegrind (instruction counts, cache miss rates) and Memcheck (no memory leaks in any version).
