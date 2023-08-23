
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

#include "Network.cuh"

#define N 20

extern __device__ func_t d_ELU;
extern __device__ func_t d_dELU;
extern __device__ func_t d_Linear;
extern __device__ func_t d_dLinear;

func_t getDeviceSymbolInGlobalMemory(func_t d_arrfunc) {
    func_t h_arrfunc;
    cudaMemcpy(&h_arrfunc, d_arrfunc, sizeof(func_t), cudaMemcpyDeviceToHost);
    return h_arrfunc;
}

func2_t getDeviceSymbolInGlobalMemory(func2_t d_func) {
    func2_t h_func;
    cudaMemcpy(&h_func, d_func, sizeof(func2_t), cudaMemcpyDeviceToHost);
    return h_func;
}

func_t d_func = 0;
func2_t d_func2 = 0;
func3_t d_func3 = 0;

int main() {

    cudaMalloc(&d_func, sizeof(func_t));

    cudaGetSymbolAddress((void**)&d_func, d_ELU);
    func_t ELU = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_dELU);
    func_t dELU = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_Linear);
    func_t Linear = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_dLinear);
    func_t dLinear = getDeviceSymbolInGlobalMemory(d_func);

    Network* n = new Network(256, 1, 4, new Layer* [4]{
        new Layer(5, ELU, dELU),
        new Layer(10, ELU, dELU),
        new Layer(5, ELU, dELU),
        new Layer(2, Linear, dLinear)
    });
    //Layer* l = new Layer(3, ELU, dELU);

    return 0;
}