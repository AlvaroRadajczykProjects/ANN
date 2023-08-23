
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cublas_v2.h>

void imprimirVectorPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
    printf("\n%s [ ", texto_mostrar);
    for (int i = inicio; i < fin; i++) {
        printf("%.8f", vector[i]);
        if (i < fin - 1) { printf(","); }
        printf(" ");
    }
    printf("]");
}

//__constant__ float lookupTable[256]; //16364
__constant__ float vector[16364]; //16364

//ncols = nelems_vectors
//nrows % num_vectors = 0
__global__ void copyVectorInMatrixRows(float* res_matrix, int nrows, int ncols, int minirows) {
    /*int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const_pos = idx / (minirows * ncols);
    int const_offset = idx % ncols;
    res_matrix[idx] = vector[const_pos * ncols + const_offset];*/
    int idx = (blockIdx.x * blockDim.x + threadIdx.x)*1024;
    for (volatile int i = 0; i < 1024; i++) {
        res_matrix[idx + i] = vector[0];
    }
}

cublasHandle_t handle;
const float alpha = 1.0f; //aparte del producto entre los elementos, puedes multiplicar esto
const float betamio = 0.0f; //aparte del producto entre los elementos, puedes multiplicar esto

const void productoMatricesDevice(const float* a, const float* b, float* c, int m, int k, int n) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &betamio, c, n);
}

int main() {

    /*
    cudaError_t cudaStatus;

    float pABC[256];

    float* h_matrix = new float[256*65536];
    float* d_matrix = 0;

    cudaMalloc(&d_matrix, 256 * 65536 * sizeof(float));

    for (int i = 0; i < 256; i++) {
        pABC[i] = i + 1;
    }

    //cudaMemcpyToSymbol(lookupTable, &pABC, sizeof(float) * 16366);
    cudaMemcpyToSymbol(vector, &pABC, sizeof(float) * 256, 0, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    copyVectorInMatrixRows <<< 16384, 1024 >>> (d_matrix, 65536, 256, 1, 2);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n\nExecution time: %f ms\n\n", time);

    //cudaMemcpy(h_matrix, d_matrix, sizeof(float) * 256 * 65535, cudaMemcpyDeviceToHost);

    //cudaMemcpyFromSymbolAsync(rABC, vector, sizeof(float) * 256, 0, cudaMemcpyDeviceToHost);

    //imprimirVectorPorPantalla("", h_matrix, 0, 512);

    cudaFree(vector);
    cudaFree(d_matrix);

    return 0;
    */

    /*

    cudaError_t cudaStatus;

    float pABC[256];

    float* h_matrix = new float[256 * 65536];
    float* d_matrix = 0;

    cudaMalloc(&d_matrix, 256 * 65536 * sizeof(float));

    for (int i = 0; i < 256; i++) {
        pABC[i] = i + 1;
    }

    //cudaMemcpyToSymbol(lookupTable, &pABC, sizeof(float) * 16366);
    cudaMemcpyToSymbol(vector, &pABC, sizeof(float) * 256, 0, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for (int i = 0; i < 65536; i++) {
        cudaMemcpyFromSymbolAsync(d_matrix + i * 256, vector, sizeof(float) * 256, 0, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n\nExecution time: %f ms\n\n", time);

    cudaMemcpy(h_matrix, d_matrix, sizeof(float) * 256 * 65536, cudaMemcpyDeviceToHost);

    //cudaMemcpyFromSymbolAsync(rABC, vector, sizeof(float) * 256, 0, cudaMemcpyDeviceToHost);

    //imprimirVectorPorPantalla("", h_matrix, 0, 512);

    cudaFree(vector);
    cudaFree(d_matrix);

    return 0;

    */

    cudaError_t cudaStatus;

    float* h_matrix = new float[8192 * 15 * 1024];
    float* d_matrix = 0;

    float* h_x = new float[8192]; //8192 X 1
    float* d_x = 0;

    for (int i = 0; i < 8192; i++) { h_x[i] = 1; }
    cudaMalloc(&d_x, 8192 * sizeof(float));
    cudaMemcpy(d_x, h_x, 8192*sizeof(float), cudaMemcpyHostToDevice);

    float* h_y = new float[1024]; //1 X 1024
    float* d_y = 0;

    for (int i = 0; i < 1024; i++) { h_y[i] = i+1; }
    cudaMalloc(&d_y, 1024 * sizeof(float));
    cudaMemcpy(d_y, h_y, 1024 * sizeof(float), cudaMemcpyHostToDevice);

    //8192 X 1024

    cudaMalloc(&d_matrix, 8192 * 15 * 1024 * sizeof(float));

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    productoMatricesDevice(d_x, d_y, d_matrix, 8192, 1, 1024);

    cublasCreate(&handle);

    //cublasSger(handle, 8192, 1024, &alpha, d_x, 8192, d_y, 1024, d_matrix, 8192);

    //cudaMemset(d_matrix, 0, 8192 * 15 * 1024 * sizeof(float));

    productoMatricesDevice(d_x, d_y, d_matrix, 8192, 1, 1024);

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\nError: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    cudaEventRecord(start, 0);

    for (int i = 0; i < 15; i++)
        productoMatricesDevice(d_x, d_y, d_matrix, 8192, 1, 1024);
    //for(int i = 0; i < 15; i++)
        //cublasSger_v2(handle, 1024, 8192, &alpha, d_x, 1, d_y, 1, d_matrix + (i * 8192 * 1024), 1024);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);

    printf("\n\nExecution time: %f ms\n\n", time);

    cudaMemcpy(h_matrix, d_matrix, sizeof(float) * 8192 * 15 * 1024, cudaMemcpyDeviceToHost);

    //cudaMemcpyFromSymbolAsync(rABC, vector, sizeof(float) * 256, 0, cudaMemcpyDeviceToHost);

    imprimirVectorPorPantalla("primero", h_matrix, 0, 2048);

    //imprimirVectorPorPantalla("segundo", h_matrix, 14*8192*1024, 14 * 8192 * 1024 + 1024);

    cudaFree(vector);
    cudaFree(d_matrix);

    return 0;

}
