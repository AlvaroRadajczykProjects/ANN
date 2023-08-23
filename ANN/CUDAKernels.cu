#include "CUDAKernels.cuh"

__global__ void applyFunctionVectorial(float* arr, func_t func) {
    //https://forums.developer.nvidia.com/t/the-float-and-float4-types-in-cuda/65061
    float4 val = reinterpret_cast<float4*>(arr)[blockIdx.x * blockDim.x + threadIdx.x];
    val.x = func(val.x);
    val.y = func(val.y);
    val.z = func(val.z);
    val.w = func(val.w);
    reinterpret_cast<float4*>(arr)[blockIdx.x * blockDim.x + threadIdx.x] = val;
}