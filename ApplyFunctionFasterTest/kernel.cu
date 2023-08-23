
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

#define N 256 * 8192 * 15
#define NITER 100

void imprimirVectorPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
    printf("\n%s [ ", texto_mostrar);
    for (int i = inicio; i < fin; i++) {
        printf("%.8f", vector[i]);
        if (i < fin - 1) { printf(","); }
        printf(" ");
    }
    printf("]");
}

using func_t = float(*) (float);

__global__ void applyFunctionScalar(float* arr) {
    arr[blockIdx.x * blockDim.x + threadIdx.x] = 3*arr[blockIdx.x * blockDim.x + threadIdx.x];
}

//no provoca que se haga más lenta la operación en absoluto!
__device__ float funcion_watelu(float x) {
    //if (x < 0) { return expf(x); }
    //else { return x; }
    return x*12345;
}
__device__ func_t p_add_func = funcion_watelu;

__device__ float funcion_wate(float x) {
    //if (x < 0) { return expf(x); }
    //else { return x; }
    return x * 54321;
}
__device__ func_t p2_add_func = funcion_wate;

__global__ void prueba(func_t* lfunc) {
    lfunc[0] = funcion_watelu;
    lfunc[1] = funcion_wate;
}

//https://forums.developer.nvidia.com/t/the-float-and-float4-types-in-cuda/65061
__global__ void applyFunctionVectorial(float* arr, func_t func) {
    float4 val = reinterpret_cast<float4*>(arr)[blockIdx.x * blockDim.x + threadIdx.x];
    val.x = func(val.x);
    val.y = func(val.y);
    val.z = func(val.z);
    val.w = func(val.w);
    reinterpret_cast<float4*>(arr)[blockIdx.x * blockDim.x + threadIdx.x] = val;
}

void xd(float* p, func_t func) {
    //func_t h_add_func;
    //cudaMemcpyFromSymbol(&h_add_func, &symbol, sizeof(func_t));
    printf("\nLo estoy haciendo en la funcion!");
    applyFunctionVectorial << < (int)ceil(N / (float)(1024 * 4)), 1024 >> > (p, func);
}

float caca(float x) {
    //if (x < 0) { return expf(x); }
    //else { return x; }
    return x * 12345;
}

func_t getDeviceSymbolInGlobalMemory(func_t d_arrfunc) {
    func_t h_arrfunc;
    cudaMemcpy(&h_arrfunc, d_arrfunc, sizeof(func_t), cudaMemcpyDeviceToHost);
    return h_arrfunc;
}

func_t d_arrfunc = 0;

int main() {

    cudaError_t cudaStatus;

    cudaMalloc(&d_arrfunc, sizeof(func_t));

    /*cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\nError A: %s", cudaGetErrorString(cudaStatus));
        return 0;
    }*/

    //func_t h_add_func;
    //cudaMemcpyFromSymbol(&h_add_func, p_add_func, sizeof(func_t));

    float* p = 0;
    cudaMalloc((void**)&p, N * sizeof(float));

    float* h_p = new float[N];
    for (int i = 0; i < N; i++) { h_p[i] = 100; }

    //imprimirVectorPorPantalla("antes", h_p, 0, N);

    cudaMemcpy(p, h_p, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time;

    cudaGetSymbolAddress((void**)&d_arrfunc, p_add_func);
    func_t h_arrfunc = getDeviceSymbolInGlobalMemory(d_arrfunc);

    cudaGetSymbolAddress((void**)&d_arrfunc, p2_add_func);
    func_t h_arrfunc2 = getDeviceSymbolInGlobalMemory(d_arrfunc);
    
    
    //printf("\nf1: %p", *h_arrfunc);

    //applyFunctionVectorial << < (int)ceil(N / (float)(1024 * 4)), 1024 >> > (p, h_arrfunc2);
    xd(p, h_arrfunc2);

    /*func_t* h_arrfunc = new func_t[2];
    func_t* d_arrfunc = 0;
    cudaMalloc(&d_arrfunc, 2 * sizeof(func_t));
    prueba << < 1, 1 >> > (d_arrfunc);
    cudaMemcpy(h_arrfunc, d_arrfunc, 2 * sizeof(func_t*), cudaMemcpyDeviceToHost);*/

    //cudaMemcpyFromSymbol(&h_arrfunc, lfunc, 2*sizeof(func_t*));

    //printf("\nprueba: %p", h_arrfunc[0]);
    //printf("\nprueba: %p", h_arrfunc[1]);
    //xd(p, h_arrfunc);

    

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //applyFunctionVectorial << < (int)ceil(N / (float)(1024 * 4)), 1024 >> > (p, h_arrfunc[1]);

    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\nError Kernel: %s", cudaGetErrorString(cudaStatus));
        return 0;
    }

    /*
    //execute kernel
    for (int i = 0; i < NITER; i++) {
        //applyFunctionScalar << < (int)ceil(N / (float)1024), 1024 >> > (p);
        //applyFunctionVectorial << < (int)ceil(N / (float)(1024 * 4)), 1024 >> > (p);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\nError");
            return 0;
        }
    }*/

    //xd(p, (const void* ) p_add_func);
    //xd(p, (const void*)funcion_watelu); //???
    //applyFunctionScalar << < (int)ceil(N / (float)1024), 1024 >> > (h_add_func, p);

    //applyFunctionVectorial << < (int)ceil(N / (float)(1024 * 4)), 1024 >> > (p, h_add_func);

    
    //cudaGetSymbolAddress((void**)&temp, p_add_func);

    //cudaGetSymbolAddress((void**)&temp, p_add_func);

    /*cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\nError 1: %s", cudaGetErrorString(cudaStatus));
        return 0;
    }

    

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\nError 2: %s", cudaGetErrorString(cudaStatus));
        return 0;
    }*/
    
    //cudaMemcpyFromSymbol(&h_add_func, "p_add_func", sizeof(func_t));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n\nExecution time: %f ms\n\n", time/(float)NITER);

    //cudaMemcpy(p, h_p, N * sizeof(float), cudaMemcpyHostToDevice);

    //applyFunctionScalar << < (int)ceil(N / (float)1024), 1024 >> > (p);
    //applyFunctionVectorial << < (int)ceil(N / (float)(1024*4)), 1024 >> > (p);

    cudaMemcpy(h_p, p, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    imprimirVectorPorPantalla("despues", h_p, 0, 20);

    cudaFree(p);
    free(h_p);

    return 0;
}

/*

// ReLU kernel using float4 data type for improved memory access
__global__ void reluKernel(const float* input, float* output, int numElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float inputValue = input[tid];
    output[tid] = inputValue;
    
}

int main() {
    int numElements = 1024*8192;  // Number of elements in the array
    size_t arraySize = numElements * sizeof(float);

    float* h_input, * h_output;  // Host arrays
    float* d_input, * d_output;  // Device arrays

    // Allocate and initialize host input array h_input

    // Allocate host output array
    h_output = (float*)malloc(arraySize);
    h_input = (float*)malloc(arraySize);

    // Allocate device memory
    cudaMalloc((void**)&d_input, arraySize);
    cudaMalloc((void**)&d_output, arraySize);

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, arraySize, cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 1024;
    int numBlocks = 8192;

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    reluKernel << <numBlocks, blockSize >> > (d_input, d_output, numElements);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n\nExecution time: %f ms\n\n", time / (float)NITER);

    // Copy the result from device to host
    cudaMemcpy(h_output, d_output, arraySize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    return 0;
}

*/