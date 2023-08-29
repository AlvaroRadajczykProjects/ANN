#include <stdio.h>
#include <cublas_v2.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <cassert>

#include <chrono>
#include <ctime>

#define uS_PER_SEC 1000000
#define uS_PER_mS 1000
#define N 4096
#define M 4096
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeCoalesced(float* odata, const float* idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void iptransposeCoalesced(float* data)
{
    __shared__ float tile_s[TILE_DIM][TILE_DIM + 1];
    __shared__ float tile_d[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (blockIdx.y > blockIdx.x) { // handle off-diagonal case
        int dx = blockIdx.y * TILE_DIM + threadIdx.x;
        int dy = blockIdx.x * TILE_DIM + threadIdx.y;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            tile_s[threadIdx.y + j][threadIdx.x] = data[(y + j) * width + x];
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            tile_d[threadIdx.y + j][threadIdx.x] = data[(dy + j) * width + dx];
        __syncthreads();
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            data[(dy + j) * width + dx] = tile_s[threadIdx.x][threadIdx.y + j];
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            data[(y + j) * width + x] = tile_d[threadIdx.x][threadIdx.y + j];
    }

    else if (blockIdx.y == blockIdx.x) { // handle on-diagonal case
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            tile_s[threadIdx.y + j][threadIdx.x] = data[(y + j) * width + x];
        __syncthreads();
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            data[(y + j) * width + x] = tile_s[threadIdx.x][threadIdx.y + j];
    }
}

int validate(const float* mat, const float* mat_t, int n, int m) {
    int result = 1;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (mat[(i * m) + j] != mat_t[(j * n) + i]) result = 0;
    return result;
}

int main() {


    float* matrix = (float*)malloc(N * M * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            matrix[(i * M) + j] = i;
    // Starting the timer

    std::chrono::time_point<std::chrono::system_clock> startCPU, endCPU;

    startCPU = std::chrono::system_clock::now();

    float* matrixT = (float*)malloc(N * M * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            matrixT[(j * N) + i] = matrix[(i * M) + j]; // matrix is obviously filled
    //Ending the timer
    
    endCPU = std::chrono::system_clock::now();

    if (!validate(matrix, matrixT, N, M)) { printf("fail!\n"); return 1; }

    std::chrono::duration<double> elapsed_seconds = endCPU - startCPU;
    std::time_t end_time = std::chrono::system_clock::to_time_t(endCPU);

    std::cout << "CPU elapsed time: " << elapsed_seconds.count() << "s; " << elapsed_seconds.count() * 1000 << "ms\n";

    float* h_matrixT, * d_matrixT, * d_matrix;
    h_matrixT = (float*)(malloc(N * M * sizeof(float)));
    cudaMalloc((void**)&d_matrixT, N * M * sizeof(float));
    cudaMalloc((void**)&d_matrix, N * M * sizeof(float));
    cudaMemcpy(d_matrix, matrix, N * M * sizeof(float), cudaMemcpyHostToDevice);

    //Starting the timer

    const float alpha = 1.0;
    const float beta = 0.0;
    cublasHandle_t handle;
    //gettimeofday(&t1, NULL);
    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, &alpha, d_matrix, M, &beta, d_matrix, N, d_matrixT, N);
    cudaDeviceSynchronize();
    startCPU = std::chrono::system_clock::now();
    for(int i = 0; i < 100; i++)
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, &alpha, d_matrix, M, &beta, d_matrix, N, d_matrixT, N);
    endCPU = std::chrono::system_clock::now();
    elapsed_seconds = endCPU - startCPU;
    end_time = std::chrono::system_clock::to_time_t(endCPU);

    std::cout << "Cublas elapsed time: " << elapsed_seconds.count()/(float)100 << "s; " << elapsed_seconds.count() / (float)100 * 1000 << "ms\n";
    cublasDestroy(handle);

    cudaMemcpy(h_matrixT, d_matrixT, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    if (!validate(matrix, h_matrixT, N, M)) { printf("fail!\n"); return 1; }
    cudaMemset(d_matrixT, 0, N * M * sizeof(float));
    memset(h_matrixT, 0, N * M * sizeof(float));
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks(N / TILE_DIM, M / TILE_DIM);
    startCPU = std::chrono::system_clock::now();
    transposeCoalesced << <blocks, threads >> > (d_matrixT, d_matrix);
    cudaDeviceSynchronize();
    endCPU = std::chrono::system_clock::now();
    cudaMemcpy(h_matrixT, d_matrixT, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    if (!validate(matrix, h_matrixT, N, M)) { printf("fail!\n"); return 1; }
    elapsed_seconds = endCPU - startCPU;
    end_time = std::chrono::system_clock::to_time_t(endCPU);

    std::cout << "Kernel out of place elapsed time: " << elapsed_seconds.count() << "s; " << elapsed_seconds.count() * 1000 << "ms\n";

    memset(h_matrixT, 0, N * M * sizeof(float));
    startCPU = std::chrono::system_clock::now();
    iptransposeCoalesced << <blocks, threads >> > (d_matrix);
    cudaDeviceSynchronize();
    endCPU = std::chrono::system_clock::now();
    cudaMemcpy(h_matrixT, d_matrix, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    if (!validate(matrix, h_matrixT, N, M)) { printf("fail!\n"); return 1; }
    elapsed_seconds = endCPU - startCPU;
    end_time = std::chrono::system_clock::to_time_t(endCPU);
    std::cout << "Kernel in place elapsed time: " << elapsed_seconds.count() << "s; " << elapsed_seconds.count() * 1000 << "ms\n";

    cudaFree(d_matrix);
    cudaFree(d_matrixT);
    return 0;
}