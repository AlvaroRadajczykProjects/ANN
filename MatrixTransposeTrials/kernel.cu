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

#include <iostream>
#include <cublas_v2.h>

void imprimirVectorPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
    printf("\n%s [ ", texto_mostrar);
    for (int i = inicio; i < fin; i++) {
        printf("%.20f", vector[i]);
        if (i < fin - 1) { printf(","); }
        printf(" ");
    }
    printf("]");
}

void imprimirMatrizPorPantalla(char* texto_mostrar, float matriz[], int n_filas, int n_columnas) {
    printf("\n%s\n", texto_mostrar);
    for (int i = 0; i < n_filas; i++) {
        imprimirVectorPorPantalla(" ", matriz, i * n_columnas, i * n_columnas + n_columnas);
    }
    printf("\n");
}

const float alpha = 1.0f;
const float beta = 0.0f;

const void productoMatricesTrasposedBDevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n) {
    cublasSgemm_v2_64(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, k, a, k, &beta, c, n);
}

const void productoMatricesTrasposedADevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n) {
    cublasSgemm_v2_64(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b, n, a, m, &beta, c, n);
}

int main() {

    /*
        IMPORTANTE SOBRE CUBLASSGEMM!! -> TODAS LAS MATRICES SE CONSIDERAN COMO COLUMN MAJOR! ES DECIR, QUE SE GUARDAN EN UN VECTOR, Y TIENEN
        CONTIGUAS LAS COLUMNAS EN VEZ DE LAS FILAS :) (MIRAR LA IMAGEN)

        CUANDO PONGO CUBLAS_OP_N, CONSIDERO LA MATRIZ COMO COLUMN MAJOR, Y CUANDO PONGO CUBLAS_OP_T, COMO ROW MAJOR (LA FORMA DE GUARDARLO EN
        UN VECTOR 1D DE TODA LA VIDA)

        EL PROBLEMA ES QUE LA MATRIZ C SIEMPRE SE VA A GUARDAR EN FORMATO COLUMN MAJOR :)

        ES POR ESO QUE A*B = C, T(A)*T(B) = T(C) -> LA TRASPUESTA DE COLUMN MAJOR ES UNA ROW MAJOR
    */

    /*
    // Perform matrix multiplication with B transposed
    // 
    // Matrix dimensions
    int m = 3; // Number of rows of A and C
    int n = 4; // Number of columns of B and C
    int k = 5; // Number of columns of A and rows of B (must match for multiplication)

    // Allocate memory for matrices on the CPU
    float* h_A = new float[m * k];
    float* h_B = new float[n * k];
    float* h_C = new float[m * n];  

    // Initialize matrices A and B
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; j++) {
            h_A[i * k + j] = i * k + j;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; j++) {
            h_B[i * k + j] = j+1;
        }
    }

    imprimirMatrizPorPantalla("A", h_A, m, k);
    imprimirMatrizPorPantalla("B", h_B, n, k);

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
    
    productoMatricesTrasposedBDevice(handle, d_A, d_B, d_C, m, k, n);
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("C", h_C, m, n);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    */

    ///*
    // Perform matrix multiplication with A transposed
    //
    // Matrix dimensions
    int m = 3; // Number of rows of A and C
    int n = 4; // Number of columns of B and C
    int k = 5; // Number of columns of A and rows of B (must match for multiplication)

    // Allocate memory for matrices on the CPU
    float* h_A = new float[k * m];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n];

    // Initialize matrices A and B
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < m; j++) {
            h_A[i * m + j] = i+1;
        }
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; j++) {
            h_B[i * n + j] = i * n + j;
        }
    }

    imprimirMatrizPorPantalla("A", h_A, k, m);
    imprimirMatrizPorPantalla("B", h_B, k, n);

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

    productoMatricesTrasposedADevice(handle, d_A, d_B, d_C, m, k, n);
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("C", h_C, m, n);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    //*/

    return 0;
}