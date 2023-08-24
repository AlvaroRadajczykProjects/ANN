﻿
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

    Layer* l1 = new Layer(2, ELU, dELU);
    Layer* l2 = new Layer(1, Linear, dLinear);

    Network* n = new Network(2, 1, 2, new Layer* [2]{
        l1,
        l2
    });

    l1->copyWeightBias(new float[4] {1, 2, 3, 4}, new float[2] {9, 8});
    l2->copyWeightBias(new float[2] {3, 4}, new float[1] {9});

    n->showInfoAboutNetwork();
    n->showWeightsBiasesLayers();

    float* res = new float[4];

    n->initForward(4);
    n->forward(4, new float[2 * 4] { 0, 0, 0, 1, 1, 0, 1, 1 }, res);
    n->finalizeForward();

    delete n;

    //Layer* l = new Layer(3, ELU, dELU);

    return 0;
}