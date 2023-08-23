#include "Layer.cuh"

#define N 10

Layer::Layer(int sz, func_t dev_act_func, func_t dev_act_der_func) {
	size = sz;
    activation_function = dev_act_func;
    activation_derivative_function = dev_act_der_func;

    /*
    cudaError_t cudaStatus;

    float* p = 0;
    cudaMalloc((void**)&p, N * sizeof(float));

    float* h_p = new float[N];
    for (int i = 0; i < N; i++) { h_p[i] = i - 9; }

    imprimirVectorPorPantalla("antes", h_p, 0, N);

    cudaMemcpy(p, h_p, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time;

    applyFunctionVectorial << < (int)ceil(N / (float)(1024 * 4)), 1024 >> > (p, activation_function);

    cudaDeviceSynchronize();

    cudaMemcpy(h_p, p, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    imprimirVectorPorPantalla("despues", h_p, 0, N);

    cudaFree(p);
    free(h_p);
    */
}

int Layer::getSize() {
    return size;
}