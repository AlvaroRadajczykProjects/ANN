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
    float4 val = reinterpret_cast<float4*>(arr)[blockIdx.x * blockDim.x + threadIdx.x];
    val.x = func(val.x);
    val.y = func(val.y);
    val.z = func(val.z);
    val.w = func(val.w);
    reinterpret_cast<float4*>(arr)[blockIdx.x * blockDim.x + threadIdx.x] = val;
}

__global__ void multiplyAllElementsByConstant(float* arr, float ct) {
    float4 val = reinterpret_cast<float4*>(arr)[blockIdx.x * blockDim.x + threadIdx.x];
    val.x = val.x * ct;
    val.y = val.y * ct;
    val.z = val.z * ct;
    val.w = val.w * ct;
    reinterpret_cast<float4*>(arr)[blockIdx.x * blockDim.x + threadIdx.x] = val;
}