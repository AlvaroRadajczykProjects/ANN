#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void imprimirVectorPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
    printf("\n%s [ ", texto_mostrar);
    for (int i = inicio; i < fin; i++) {
        printf("%.8f", vector[i]);
        if (i < fin - 1) { printf(","); }
        printf(" ");
    }
    printf("]");
}

const size_t ds = 1024;
const size_t rows = 8192;
const int nTPB = 1024;

__constant__ float vector[ds]; //16364

__global__ void k(float* dout) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    float my_val;
    if (idx < ds) {
        my_val = vector[idx];
        for (size_t i = 0; i < rows; i++) {
            dout[idx] = my_val;
            idx += ds;
        }
    }
}

int main() {


    float* d_in = 0;
    float* d_out = 0;
    float h_in[ds];
    float* h_out = new float[ds * rows];
    for (int i = 0; i < ds; i++) h_in[i] = i+1;

    cudaMalloc(&d_out, ds * rows * sizeof(float));

    cudaMemcpyToSymbol(vector, &h_in, sizeof(float) * ds);

    //cudaMemcpy(d_in, h_in, ds * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for(int i = 0; i < 16; i++)
        k << <(ds + nTPB - 1) / nTPB, nTPB >> > (d_out);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n\nExecution time: %f ms\n\n", time);

    cudaMemcpy(h_out, d_out, ds * rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    imprimirVectorPorPantalla("", h_out, 8192-1024, 8192);
    
}