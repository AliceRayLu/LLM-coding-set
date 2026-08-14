// GEMM GPU kernels for benchmarking against cuBLAS.
//
// Compiled from the notebook (1.1-GEMM.ipynb) as a shared library:
//   nvcc -O3 -arch=sm_80 -Xcompiler -fPIC -shared -o gemm_gpu.so gemm_gpu.cu
//
// All kernels use row-major layout:
//   C[i*N+j] = sum_k A[i*K+k] * B[k*N+j]
// Kernels are __global__; Python calls them through the extern "C" host
// launchers below, which configure grid/block and launch on a
// caller-provided stream (so CUDA-event timing works).

#include <cuda_runtime.h>

// ─── Naive GEMM ──────────────────────────────────────────────
// One thread per output element, no shared memory, no tiling.
__global__ void sgemm_naive_kernel(
    const int M, const int N, const int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tuning parameters
constexpr int NAIVE_BLOCK = 16;  // 16×16 = 256 threads/block

extern "C" void sgemm_naive_launch(
    const int M, const int N, const int K,
    const float* A, const float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(NAIVE_BLOCK, NAIVE_BLOCK);
    const dim3 grid(
        (N + NAIVE_BLOCK - 1) / NAIVE_BLOCK,
        (M + NAIVE_BLOCK - 1) / NAIVE_BLOCK);
    sgemm_naive_kernel<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}

// ─── Add more kernels here ───────────────────────────────────
// Pattern: write the __global__ kernel, then an extern "C" launcher
// that takes (M, N, K, A, B, C, cudaStream_t) and launches it.
// The notebook registers each launcher via _make_cu_wrapper.

#define TILE_SIZE 16

// every block deals with one tile
__global__ void sgemm_tile_kernel(
    const int M, const int N, const int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C){
    __shared__ float a[TILE_SIZE][TILE_SIZE];
    __shared__ float b[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    const int num_tile = (K+TILE_SIZE-1) / TILE_SIZE;
    float sum = 0.0;
    for(int i = 0;i < num_tile;i++){
        const int r = i*TILE_SIZE + threadIdx.y;
        const int c = i*TILE_SIZE + threadIdx.x;

        a[threadIdx.y][threadIdx.x] = (row < M && c < K) ? A[row*K+c]:0.0f;
        b[threadIdx.y][threadIdx.x] = (r < K && col < N) ? B[r*N+col]:0.0f;
        __syncthreads();

        for(int j = 0;j < TILE_SIZE;j++){
            sum += a[threadIdx.y][j] * b[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M && col < N) C[row*N+col] = sum;
}

extern "C" void sgemm_tile_launch(
    const int M, const int N, const int K,
    const float* A, const float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    sgemm_tile_kernel<<<grid, block,0, stream>>>(M,N,K,A,B,C);
}


__global__ void sgemm_tensor_core_kernel(){
    
}