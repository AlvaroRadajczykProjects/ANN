#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>

#include <cublas_v2.h>

#include <iostream>
#include <chrono>
#include <ctime>

#include <stdio.h>

void imprimirVectorIntPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
    printf("\n%s [ ", texto_mostrar);
    for (int i = inicio; i < fin; i++) {
        printf("%.8f", vector[i]);
        if (i < fin - 1) { printf(","); }
        printf(" ");
    }
    printf("]");
}

void imprimirMatrizPorPantalla(char* texto_mostrar, float matriz[], int n_filas, int n_columnas) {
    printf("\n%s\n", texto_mostrar);
    for (int i = 0; i < n_filas; i++) {
        imprimirVectorIntPorPantalla(" ", matriz, i * n_columnas, i * n_columnas + n_columnas);
    }
    printf("\n");
}

int main() {
    // Matrix dimensions
    int m = 3; // Number of rows of A and C
    int n = 4; // Number of columns of B and C
    int k = 5; // Number of columns of A and rows of B (must match for multiplication)

    // Allocate memory for matrices on the CPU
    float* h_A = new float[m * k];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n];

    // Initialize matrices A and B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_A[i * k + j] = 1;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            h_B[i * k + j] = i + 1;
        }
    }

    imprimirMatrizPorPantalla("", h_A, m, k);
    imprimirMatrizPorPantalla("", h_B, n, k);

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory for matrices on the GPU
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Transfer data from CPU to GPU
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication with B transposed
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, d_A, m, d_B, n, &beta, d_C, m);

    // Transfer the result back to the CPU
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix
    imprimirMatrizPorPantalla("", h_C, m, n);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
