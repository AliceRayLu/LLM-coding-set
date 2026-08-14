// GEMM CPU implementations for benchmarking against cuBLAS.
//
// Compiled from the notebook (1.1-GEMM.ipynb) as a shared library:
//   g++ -O3 -march=native -fopenmp -shared -fPIC -o libgemm_cpu.so gemm_cpu.cpp
//
// All functions use row-major layout:
//   C[i*N+j] = sum_k A[i*K+k] * B[k*N+j]
// C must be zero-initialized by the caller: the ikj / tiled / omp variants
// accumulate into it (naive assigns, so it does not rely on zeroing).

#include <algorithm>

extern "C" {

// ─── Naive ijk ────────────────────────────────────────────────
// Inner loop reads B with stride N — worst cache behaviour.
void sgemm_naive(const int M, const int N, const int K,
                 const float* __restrict__ A,
                 const float* __restrict__ B,
                 float* __restrict__ C) {
    for(int i = 0;i < M;i++){
        for(int j = 0;j < N;j++){
            for(int k = 0;k < K;k++){
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// ─── ikj: loop reordering ─────────────────────────────────────
// Inner loop over j reads B row-wise; the compiler auto-vectorizes it.
void sgemm_ikj(const int M, const int N, const int K,
               const float* __restrict__ A,
               const float* __restrict__ B,
               float* __restrict__ C) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i*N+j] += A[i*K+k] * B[k*N+j];
            }
        }
    }
}

// ─── Tiled ikj: cache blocking on 64×64 tiles ─────────────────
void sgemm_tiled(const int M, const int N, const int K,
                 const float* __restrict__ A,
                 const float* __restrict__ B,
                 float* __restrict__ C) {
    constexpr int TILE = 64;
    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int k0 = 0; k0 < K; k0 += TILE) {
            for (int j0 = 0; j0 < N; j0 += TILE) {
                // inner loop
                for (int i = i0; i < std::min(i0+TILE, M); ++i) {
                    for (int k = k0; k < std::min(k0+TILE, K); ++k) {
                        for (int j = j0; j < std::min(j0+TILE, N); ++j) {
                            C[i*N+j] += A[i*K+k] * B[k*N+j];
                        }
                    }
                }
            }
        }
    }
}

// ─── OpenMP: parallelize rows of the ikj version ──────────────
// Compiled only when the library is built with -fopenmp.
#ifdef _OPENMP
void sgemm_omp(const int M, const int N, const int K,
               const float* __restrict__ A,
               const float* __restrict__ B,
               float* __restrict__ C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        float* ci = C + i * N;
        const float* ai = A + i * K;
        for (int k = 0; k < K; ++k) {
            const float aik = ai[k];
            const float* bk = B + k * N;
            for (int j = 0; j < N; ++j) {
                ci[j] += aik * bk[j];
            }
        }
    }
}
#endif

}  // extern "C"
