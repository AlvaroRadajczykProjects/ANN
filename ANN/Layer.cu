#include "Layer.cuh"

#define N 10

using namespace std;

Layer::Layer(int sz, func_t dev_act_func, func_t dev_act_der_func) {
	size = sz;
    activation_function = dev_act_func;
    activation_derivative_function = dev_act_der_func;
}

Layer::~Layer() {
    handle = NULL;
}

void Layer::showInfo() {
    printf("\n\t\tInput size: %d", input_size);
    printf("\n\t\tSize: %d", size);
    printf("\n\t\tNumber of networks: %d", number_networks);
    printf("\n\t\tFirst layer?: %s", is_first_layer ? "Yes" : "No");
    printf("\n\t\tIs training?: %s", is_training ? "Yes" : "No");
    printf("\n");
}

void Layer::showWeightBias() {
    for (int i = 0; i < number_networks; i++) {
        float* h_weight_m = new float[input_size * size];
        float* h_bias_v = new float[size];
        cudaMemcpy(h_weight_m, hd_weight_matrices_pointers[i], input_size * size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bias_v, hd_bias_vectors_pointers[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tbias:", h_bias_v, 1, size);
        imprimirMatrizPorPantalla("\n\t\tweight:", h_weight_m, input_size, size);
        delete h_weight_m;
        delete h_bias_v;
    }
}

void Layer::showAuxiliarExpandReduce() {
    for (int i = 0; i < number_networks; i++) {
        float* h_auxiliar_expand_reduce_matrix = new float[number_input_examples];
        cudaMemcpy(h_auxiliar_expand_reduce_matrix, hd_expand_reduce_matrix_pointers[i], number_input_examples * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tauxiliar:", h_auxiliar_expand_reduce_matrix, 1, size);
        delete h_auxiliar_expand_reduce_matrix;
    }
}

void Layer::showForward() {
    for (int i = 0; i < number_networks; i++) {
        float* h_forward = new float[number_input_examples * size];
        cudaMemcpy(h_forward, hd_forward_pointers[i], number_input_examples * size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tforward matrix:", h_forward, number_input_examples, size);
        delete h_forward;
    }
}

int Layer::getSize() {
    return size;
}

float* Layer::getDeviceForward() {
    return d_forward;
}

float** Layer::getDeviceForwardPointers() {
    return d_forward_pointers;
}

void Layer::setMaxNumThreads(int set) {
    max_num_threads = set;
}

void Layer::setInputSize(int is) {
    input_size = is;
}

void Layer::setNumberInputExamples(int set) {
    number_input_examples = set;
}

void Layer::setAuxiliarExpandReduceMatrix(float* set) {
    d_auxiliar_expand_reduce_matrix = set;
    if (set != NULL) {
        hd_expand_reduce_matrix_pointers = new float* [number_networks];
        cudaMalloc(&d_expand_reduce_matrix_pointers, number_networks * sizeof(float*));
        for (int i = 0; i < number_networks; i++) { hd_expand_reduce_matrix_pointers[i] = d_auxiliar_expand_reduce_matrix; }
        cudaMemcpy(d_expand_reduce_matrix_pointers, hd_expand_reduce_matrix_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    } else {
        cudaFree(d_expand_reduce_matrix_pointers);
        delete hd_expand_reduce_matrix_pointers;
    }
}

void Layer::setNumberNetworks(int nn) {
    number_networks = nn;
}

void Layer::setIsFirstLayer(bool set) {
    is_first_layer = set;
}

void Layer::setIsTraining(bool set) {
    is_training = set;
}

void Layer::setCublasHandle(cublasHandle_t* h) {
    handle = h;
}

void Layer::forward(cudaStream_t stream, float** d_input_pointers) {
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_bias_vectors_pointers, d_forward_pointers, number_input_examples, 1, size, number_networks);
    productoMatricesBatchDeviceSumC(*handle, d_input_pointers, d_weight_matrices_pointers, d_forward_pointers, number_input_examples, input_size, size, number_networks);
    applyFunctionVectorial <<< num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_function);
}

void Layer::forward(cudaStream_t stream, Layer* previous_layer) {
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_bias_vectors_pointers, d_forward_pointers, number_input_examples, 1, size, number_networks);
    productoMatricesBatchDeviceSumC(*handle, previous_layer->getDeviceForwardPointers(), d_weight_matrices_pointers, d_forward_pointers, number_input_examples, input_size, size, number_networks);
    applyFunctionVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_function);
}

void Layer::allocWeightMatricesMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaMalloc( &d_array_weight_matrix, input_size * size * number_networks * sizeof(float));
        cudaMalloc( &d_array_bias_vector, size * number_networks * sizeof(float));
        hd_weight_matrices_pointers = new float* [number_networks];
        hd_bias_vectors_pointers = new float* [number_networks];
        cudaMalloc(&d_weight_matrices_pointers, number_networks * sizeof(float*));
        cudaMalloc(&d_bias_vectors_pointers, number_networks * sizeof(float*));
        for (int i = 0; i < number_networks; i++) {
            hd_weight_matrices_pointers[i] = d_array_weight_matrix + i*(input_size * size);
            hd_bias_vectors_pointers[i] = d_array_bias_vector + i * (size);
        }
        cudaMemcpy(d_weight_matrices_pointers, hd_weight_matrices_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias_vectors_pointers, hd_bias_vectors_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    }
}

void Layer::freeWeightMatricesMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaFree(d_array_weight_matrix); d_array_weight_matrix = NULL;
        cudaFree(d_array_bias_vector); d_array_bias_vector = NULL;
        delete hd_weight_matrices_pointers;  hd_weight_matrices_pointers = NULL;
        delete hd_bias_vectors_pointers; hd_bias_vectors_pointers = NULL;
        cudaFree(d_weight_matrices_pointers); d_weight_matrices_pointers = NULL;
        cudaFree(d_bias_vectors_pointers); d_bias_vectors_pointers = NULL;
    }
    input_size = 0;
    size = 0;
    number_networks = 0;
}

void Layer::allocForwardMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0 && number_input_examples > 0) {
        num_blocks_needed_apply_function = (int)ceil((size * number_input_examples) / (float)(max_num_threads * 4));
        num_threads_needed_apply_function = min(max_num_threads, number_input_examples * size * 4);

        cudaMalloc(&d_forward, nextFourMultiple( number_input_examples * size * number_networks ) * sizeof(float));
        hd_forward_pointers = new float* [number_networks];
        cudaMalloc(&d_forward_pointers, number_networks * sizeof(float*));
        for (int i = 0; i < number_networks; i++) {
            hd_forward_pointers[i] = d_forward + ( i * number_input_examples * size);
        }
        cudaMemcpy(d_forward_pointers, hd_forward_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    }
}

void Layer::freeForwardMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0 && number_input_examples > 0) {
        num_blocks_needed_apply_function = 0;
        num_threads_needed_apply_function = 0;

        cudaFree(d_forward); d_forward = NULL;
        delete hd_forward_pointers; hd_forward_pointers = NULL;
        cudaFree(d_forward_pointers); d_forward_pointers = NULL;
    }
    number_input_examples = 0;
}

void Layer::copyWeightBias(float* h_weight, float* h_bias) {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaMemcpy(d_array_weight_matrix, h_weight, input_size * size * number_networks * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array_bias_vector, h_bias, size * number_networks * sizeof(float), cudaMemcpyHostToDevice);
    }
}