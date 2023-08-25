#include "CUDAKernels.cuh"

void manageCUDAError(cudaError_t status, char* description) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Error de CUDA %s: %s\n", description, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

int nextFourMultiple(int val) {
    if (val % 4 == 0) { return val; }
    else { return val + (4 - (val % 4)); }
}

const void productoMatricesDevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n) {
    cublasSgemm_v2_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta_nosum, c, n);
}

const void productoMatricesBatchDevice(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr) {
    cublasSgemmBatched_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta_nosum, c, n, num_matr);
}

const void productoMatricesBatchDeviceSumC(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr) {
    cublasSgemmBatched_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta_sum, c, n, num_matr);
}

__global__ void applyFunctionVectorial(float* arr, func_t func) {
    //https://forums.developer.nvidia.com/t/the-float-and-float4-types-in-cuda/65061
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = reinterpret_cast<float4*>(arr)[idx];
    val.x = func(val.x);
    val.y = func(val.y);
    val.z = func(val.z);
    val.w = func(val.w);
    reinterpret_cast<float4*>(arr)[idx] = val;
}

__global__ void applyLossFunctionVectorial(float* pred, float* real, float* dst, func2_t func) {
    //https://forums.developer.nvidia.com/t/the-float-and-float4-types-in-cuda/65061
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 vpred = reinterpret_cast<float4*>(pred)[idx];
    float4 vreal = reinterpret_cast<float4*>(real)[idx];
    vpred.x = func(vpred.x, vreal.x);
    vpred.y = func(vpred.y, vreal.y);
    vpred.z = func(vpred.z, vreal.z);
    vpred.w = func(vpred.w, vreal.w);
    reinterpret_cast<float4*>(dst)[idx] = vpred;
}

__global__ void multiplyAllElementsByConstant(float* arr, float ct) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = reinterpret_cast<float4*>(arr)[idx];
    val.x = val.x * ct;
    val.y = val.y * ct;
    val.z = val.z * ct;
    val.w = val.w * ct;
    reinterpret_cast<float4*>(arr)[idx] = val;
}

__global__ void sumVectorsSameDimensions(float* dst, float* src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val_src = reinterpret_cast<float4*>(src)[idx];
    float4 val_dst = reinterpret_cast<float4*>(dst)[idx];
    val_src.x = val_src.x + val_dst.x;
    val_src.y = val_src.y + val_dst.y;
    val_src.z = val_src.z + val_dst.z;
    val_src.w = val_src.w + val_dst.w;
    reinterpret_cast<float4*>(dst)[idx] = val_src;
}