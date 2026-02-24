#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>


#define chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
   
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
           
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}


template <int TS>
__global__ void sgemm_tiled(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    float tmp = 0.0;

   
    for (int k_offset = 0; k_offset < K; k_offset += TS) {
        
        int row_a = block_row * TS + thread_row;
        int col_a = k_offset + thread_col;
        if (row_a < M && col_a < K)
            As[thread_row][thread_col] = A[row_a * K + col_a];
        else
            As[thread_row][thread_col] = 0.0f;

        int row_b = k_offset + thread_row;
        int col_b = block_col * TS + thread_col;
        if (row_b < K && col_b < N)
            Bs[thread_row][thread_col] = B[row_b * N + col_b];
        else
            Bs[thread_row][thread_col] = 0.0f;

        __syncthreads(); 

       
        for (int i = 0; i < TS; ++i) {
            tmp += As[thread_row][i] * Bs[i][thread_col];
        }
        
        __syncthreads();
    }

    int row = block_row * TS + thread_row;
    int col = block_col * TS + thread_col;
    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}


void run_benchmark(int N, bool use_tiled) {
    size_t size = (size_t)N * N * sizeof(float);
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

    const int TS = 32; 
    dim3 threads(TS, TS);
    dim3 blocks((N + TS - 1) / TS, (N + TS - 1) / TS);

   
    for(int i = 0; i < 5; i++) {
        if(use_tiled) sgemm_tiled<TS><<<blocks, threads, 0, stream>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
        else sgemm_naive<<<blocks, threads, 0, stream>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
    }
    chk(cudaStreamSynchronize(stream));

   
    cudaEvent_t start, stop;
    chk(cudaEventCreate(&start));
    chk(cudaEventCreate(&stop));
    chk(cudaEventRecord(start, stream));

    const int ITERS = 50;
    for(int i = 0; i < ITERS; i++) {
        if(use_tiled) sgemm_tiled<TS><<<blocks, threads, 0, stream>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
        else sgemm_naive<<<blocks, threads, 0, stream>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
    }

    chk(cudaEventRecord(stop, stream));
    chk(cudaEventSynchronize(stop));

    float ms = 0.0f;
    chk(cudaEventElapsedTime(&ms, start, stop));
    float avg_us = (ms * 1000.0f) / ITERS;

  
    double flops_per_matmul = 2.0 * double(N) * double(N) * double(N);
    double tflops = (flops_per_matmul / (avg_us * 1e-6)) * 1e-12;

    printf("[%s] Size: %d x %d, Avg Time: %.2f us, Performance: %.2f TFLOPS\n", 
           use_tiled ? "TILED" : "NAIVE", N, N, avg_us, tflops);

    
    chk(cudaEventDestroy(start)); chk(cudaEventDestroy(stop));
    chk(cudaStreamDestroy(stream));
    chk(cudaFree(d_A)); chk(cudaFree(d_B)); chk(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
}

int main() {
  
    int sizes[] = {1024, 2048, 4096};
    for(int n : sizes) {
        run_benchmark(n, false); 
        run_benchmark(n, true);  
        printf("------------------------------------------------------------\n");
    }
    return 0;
}
