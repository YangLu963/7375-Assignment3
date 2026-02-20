#include <cstdio>
#include <cuda_runtime.h>


#define chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

template <int BK, int BN, int BM>
__global__ void sgemm_tiled(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    float tmp = 0.0;

  
    for (int k_offset = 0; k_offset < K; k_offset += BK) {
        As[thread_row][thread_col] = A[(block_row * BM + thread_row) * K + (k_offset + thread_col)];
        Bs[thread_row][thread_col] = B[(k_offset + thread_row) * N + (block_col * BN + thread_col)];
        
        __syncthreads(); 

     
        for (int i = 0; i < BK; ++i) {
            tmp += As[thread_row][i] * Bs[i][thread_col];
        }
        
        __syncthreads(); 
    }

    int row = block_row * BM + thread_row;
    int col = block_col * BN + thread_col;
    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}


void run_benchmark(int N, bool use_tiled) {
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    chk(cudaMalloc(&d_A, size));
    chk(cudaMalloc(&d_B, size));
    chk(cudaMalloc(&d_C, size));

    for(int i=0; i<N*N; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; h_C[i] = 0.0f; }
    chk(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    chk(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    chk(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    chk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const int TS = 32; // Block Size
    dim3 threads(TS, TS);
    dim3 blocks((N + TS - 1) / TS, (N + TS - 1) / TS);

    for(int i = 0; i < 10; i++) {
        if(use_tiled) sgemm_tiled<TS, TS, TS><<<blocks, threads, 0, stream>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
        else sgemm_naive<<<blocks, threads, 0, stream>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
    }
    chk(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    chk(cudaEventCreate(&start));
    chk(cudaEventCreate(&stop));
    chk(cudaEventRecord(start, stream));

    const int ITERS = 50;
    for(int i = 0; i < ITERS; i++) {
        if(use_tiled) sgemm_tiled<TS, TS, TS><<<blocks, threads, 0, stream>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
        else sgemm_naive<<<blocks, threads, 0, stream>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
    }

    chk(cudaEventRecord(stop, stream));
    chk(cudaEventSynchronize(stop));

    float ms = 0.0f;
    chk(cudaEventElapsedTime(&ms, start, stop));
    float avg_us = (ms * 1000.0f) / ITERS;

    double flops_per_matmul = 2.0 * double(N) * double(N) * double(N);
    double tflops = (flops_per_matmul / (avg_us * 1e-6)) * 1e-12;

    printf("[%s] Size: %d, Time: %.2f us, Performance: %.2f TFLOPS\n", 
           use_tiled ? "TILED" : "NAIVE", N, avg_us, tflops);

    chk(cudaEventDestroy(start)); chk(cudaEventDestroy(stop));
    chk(cudaStreamDestroy(stream));
    chk(cudaFree(d_A)); chk(cudaFree(d_B)); chk(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
}

int main() {
    int N = 4096; 
    run_benchmark(N, false); 
    run_benchmark(N, true);  
    return 0;
}
