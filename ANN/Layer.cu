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

Layer::~Layer() {
    handle = NULL;
}

void Layer::showInfo() {
    printf("\n\t\tInput size: %d", input_size);
    printf("\n\t\tSize: %d", size);
    printf("\n\t\tNumber of networks: %d", number_networks);
    printf("\n");
}

void Layer::showWeightBias() {
    float* h_weight_m = new float[input_size * size * number_networks];
    float* h_bias_v = new float[size * number_networks];
    cudaMemcpy(h_weight_m, d_array_weight_matrix, input_size * size * number_networks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bias_v, d_array_bias_vector, size * number_networks * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < number_networks; i++) {
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tbias:", h_bias_v + (i * size), 1, size);
        imprimirMatrizPorPantalla("\n\t\tweight:", h_weight_m + (i * input_size * size), input_size, size);
    }
    delete h_weight_m;
    delete h_bias_v;
}

int Layer::getSize() {
    return size;
}

void Layer::setInputSize(int is) {
    input_size = is;
}

void Layer::setNumberNetworks(int nn) {
    number_networks = nn;
}

void Layer::setCublasHandle(cublasHandle_t* h) {
    handle = h;
}

void Layer::allocMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaMalloc( &d_array_weight_matrix, input_size * size * number_networks * sizeof(float));
        cudaMalloc( &d_array_bias_vector, size * number_networks * sizeof(float));
    }
}

void Layer::freeMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaFree(d_array_weight_matrix); d_array_weight_matrix = NULL;
        cudaFree(d_array_bias_vector); d_array_bias_vector = NULL;
    }
    input_size = 0;
    size = 0;
    number_networks = 0;
}

void Layer::copyWeightBias(float* h_weight, float* h_bias) {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaMemcpy(d_array_weight_matrix, h_weight, input_size * size * number_networks * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array_bias_vector, h_bias, size * number_networks * sizeof(float), cudaMemcpyHostToDevice);
    }
}