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

void Layer::showErrorWeightBias() {
    for (int i = 0; i < number_networks; i++) {
        float* h_weight_m = new float[input_size * size];
        float* h_bias_v = new float[size];
        cudaMemcpy(h_weight_m, hd_error_weight_matrices_pointers[i], input_size * size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bias_v, hd_error_bias_vectors_pointers[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tbias:", h_bias_v, 1, size);
        imprimirMatrizPorPantalla("\n\t\tweight:", h_weight_m, input_size, size);
        delete h_weight_m;
        delete h_bias_v;
    }
}

void Layer::showAuxiliarExpandReduce() {
    for (int i = 0; i < number_networks; i++) {
        float* h_auxiliar_expand_reduce_matrix = new float[size];
        cudaMemcpy(h_auxiliar_expand_reduce_matrix, hd_expand_reduce_matrix_pointers[i], size * sizeof(float), cudaMemcpyDeviceToHost);
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


float** Layer::getAuxiliarExpandReduceMatrixPointers() {
    return d_expand_reduce_matrix_pointers;
}

float** Layer::getDeviceAuxiliarErrorForwardLayerPointers() {
    return d_auxiliar_error_forward_layer_pointers;
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

void Layer::initWeightBiasValues(curandGenerator_t curandGenerator) {
    cudaMemset(d_array_bias_vector, 0, nextFourMultiple(size * number_networks) * sizeof(float));
    unsigned long long semilla = rand() % 10000;
    curandSetPseudoRandomGeneratorSeed(curandGenerator, semilla);
    curandGenerateNormal(curandGenerator, (float*)d_array_weight_matrix, nextFourMultiple(input_size * size * number_networks), 0.0, 2.0/(float)input_size );
    cudaDeviceSynchronize();
}

void Layer::forward(cudaStream_t stream, float** d_input_pointers, int num_inputs) {
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_bias_vectors_pointers, d_forward_pointers, num_inputs, 1, size, number_networks);
    productoMatricesBatchDeviceSumC(*handle, d_input_pointers, d_weight_matrices_pointers, d_forward_pointers, num_inputs, input_size, size, number_networks);
    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size*num_inputs*number_networks), d_forward, activation_function);
    //applyFunctionVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_function);
}

void Layer::forward(cudaStream_t stream, Layer* previous_layer, int num_inputs) {
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_bias_vectors_pointers, d_forward_pointers, num_inputs, 1, size, number_networks);
    productoMatricesBatchDeviceSumC(*handle, previous_layer->getDeviceForwardPointers(), d_weight_matrices_pointers, d_forward_pointers, num_inputs, input_size, size, number_networks);
    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_inputs * number_networks), d_forward, activation_function);
    //applyFunctionVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_function);
}

void Layer::backward(cudaStream_t stream, Layer* previous_layer, int num_outputs) {

    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);
    
    managedMultiplyMatricesSameDimensions(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, d_forward);

    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, 1 / (float)num_outputs);

    //float* h_res = new float[nextFourMultiple(max(input_size, size) * num_outputs * number_networks)];
    //cudaMemcpy(h_res, d_auxiliar_error_forward_layer, nextFourMultiple(size * num_outputs * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    //imprimirMatrizPorPantalla("bias nada mas ser calculado: ", h_res, num_outputs * number_networks, size);

    //bias error
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_auxiliar_error_forward_layer_pointers, d_error_bias_vectors_pointers, 1, num_outputs, size, number_networks);

    //weight error
    productoMatricesTrasposedABatchDevice(*handle, previous_layer->getDeviceForwardPointers(), d_auxiliar_error_forward_layer_pointers, d_error_weight_matrices_pointers, input_size, num_outputs, size, number_networks);

    //previous layer error
    productoMatricesTrasposedBBatchDevice(*handle, d_auxiliar_error_forward_layer_pointers, d_weight_matrices_pointers, d_auxiliar2_error_forward_layer_pointers, num_outputs, size, input_size, number_networks);
    cudaMemcpy(d_auxiliar_error_forward_layer, d_auxiliar2_error_forward_layer, nextFourMultiple(num_outputs * input_size * number_networks) * sizeof(float), cudaMemcpyDeviceToDevice);
    
    /*
    //applyFunctionVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_derivative_function);
    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);
    //multiplyMatricesSameDimensionsVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_auxiliar_error_forward_layer, d_forward);
    managedMultiplyMatricesSameDimensions( stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, d_forward);
    //multiplyAllElementsByConstantVectorial << < (int) ceil(nextFourMultiple(num_outputs) / ((float)max_num_threads * 4)), min(max_num_threads, (int)(nextFourMultiple(number_networks * size) / (float)4)), 0, stream >> > (d_error_array_bias_vector, 1 / (float)num_outputs);
    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, 1 / (float)num_outputs);
    //bias error
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_auxiliar_error_forward_layer_pointers, d_error_bias_vectors_pointers, 1, num_outputs, size, number_networks);
    //weight error
    productoMatricesTrasposedABatchDevice(*handle, previous_layer->getDeviceForwardPointers(), d_auxiliar_error_forward_layer_pointers, d_error_weight_matrices_pointers, input_size, num_outputs, size, number_networks);
    //previous layer error
    productoMatricesTrasposedBBatchDevice(*handle, d_auxiliar_error_forward_layer_pointers, d_weight_matrices_pointers, d_auxiliar2_error_forward_layer_pointers, num_outputs, size, input_size, number_networks);
    cudaMemcpy(d_auxiliar_error_forward_layer, d_auxiliar2_error_forward_layer, nextFourMultiple(num_outputs * input_size * number_networks) * sizeof(float), cudaMemcpyDeviceToDevice);
    */
}

void Layer::backward(cudaStream_t stream, float** input_pointers, int num_outputs) {
    
    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);

    managedMultiplyMatricesSameDimensions(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, d_forward);

    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, 1 / (float)num_outputs);

    //bias error
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_auxiliar_error_forward_layer_pointers, d_error_bias_vectors_pointers, 1, num_outputs, size, number_networks);

    //weight error
    productoMatricesTrasposedABatchDevice(*handle, input_pointers, d_auxiliar_error_forward_layer_pointers, d_error_weight_matrices_pointers, input_size, num_outputs, size, number_networks);

    //float* h_res = new float[nextFourMultiple(max(input_size, size) * num_outputs * number_networks)];
    //cudaMemcpy(h_res, d_auxiliar_error_forward_layer, nextFourMultiple(input_size * num_outputs * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    //imprimirMatrizPorPantalla("error de la capa posterior calculado: ", h_res, num_outputs * number_networks, input_size);
    
    //managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);

    /*
    //applyFunctionVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_derivative_function);
    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);
    //multiplyMatricesSameDimensionsVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_auxiliar_error_forward_layer, d_forward);
    managedMultiplyMatricesSameDimensions(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, d_forward);
    //multiplyAllElementsByConstantVectorial << < (int)ceil(nextFourMultiple(num_outputs) / ((float)max_num_threads * 4)), min(max_num_threads, (int)(nextFourMultiple(number_networks * size) / (float)4)), 0, stream >> > (d_error_array_bias_vector, 1 / (float)num_outputs);
    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, 1 / (float)num_outputs);
    //bias error
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_auxiliar_error_forward_layer_pointers, d_error_bias_vectors_pointers, 1, num_outputs, size, number_networks);
    //weight error
    productoMatricesTrasposedABatchDevice(*handle, input_pointers, d_auxiliar_error_forward_layer_pointers, d_error_weight_matrices_pointers, input_size, num_outputs, size, number_networks);
    */
}

void Layer::applyGradientSGD(cudaStream_t stream, float lrate) {
    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(input_size * size * number_networks), d_error_array_weight_matrix, -lrate);
    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * number_networks), d_error_array_bias_vector, -lrate);
    managedSumVectorsSameDimensions(stream, max_num_threads, nextFourMultiple(input_size * size * number_networks), d_array_weight_matrix, d_error_array_weight_matrix);
    managedSumVectorsSameDimensions(stream, max_num_threads, nextFourMultiple(size * number_networks), d_array_bias_vector, d_error_array_bias_vector);
}

void Layer::allocWeightMatricesMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaMalloc( &d_array_weight_matrix, nextFourMultiple(input_size * size * number_networks) * sizeof(float));
        cudaMalloc( &d_array_bias_vector, nextFourMultiple(size * number_networks) * sizeof(float));
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

void Layer::allocBackwardMemory(int batch_size, float* d_aux_error_matrix, float* d_aux2_error_matrix) {
    d_auxiliar_error_forward_layer = d_aux_error_matrix;
    d_auxiliar2_error_forward_layer = d_aux2_error_matrix;

    cudaMalloc( &d_error_array_weight_matrix, nextFourMultiple(input_size * size * number_networks) * sizeof(float));
    cudaMalloc( &d_error_array_bias_vector, nextFourMultiple(size * number_networks) * sizeof(float));
    hd_error_weight_matrices_pointers = new float* [number_networks];
    hd_error_bias_vectors_pointers = new float* [number_networks];
    float** hd_auxiliar_error_forward_layer_pointers = new float* [number_networks];
    float** hd_auxiliar2_error_forward_layer_pointers = new float* [number_networks];
    cudaMalloc(&d_auxiliar_error_forward_layer_pointers, number_networks * sizeof(float*));
    cudaMalloc(&d_auxiliar2_error_forward_layer_pointers, number_networks * sizeof(float*));
    cudaMalloc(&d_error_weight_matrices_pointers, number_networks * sizeof(float*));
    cudaMalloc(&d_error_bias_vectors_pointers, number_networks * sizeof(float*));
    for (int i = 0; i < number_networks; i++) {
        hd_error_weight_matrices_pointers[i] = d_error_array_weight_matrix + i*(input_size * size);
        hd_error_bias_vectors_pointers[i] = d_error_array_bias_vector + i * (size);
        hd_auxiliar_error_forward_layer_pointers[i] = d_auxiliar_error_forward_layer + i * (size * batch_size);
        hd_auxiliar2_error_forward_layer_pointers[i] = d_auxiliar2_error_forward_layer + i * (size * batch_size);
    }
    cudaMemcpy(d_error_weight_matrices_pointers, hd_error_weight_matrices_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_error_bias_vectors_pointers, hd_error_bias_vectors_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_auxiliar_error_forward_layer_pointers, hd_auxiliar_error_forward_layer_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_auxiliar2_error_forward_layer_pointers, hd_auxiliar2_error_forward_layer_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    delete hd_auxiliar_error_forward_layer_pointers;
    delete hd_auxiliar2_error_forward_layer_pointers;
}

void Layer::freeBackwardMemory() {
    d_auxiliar_error_forward_layer = NULL;
    d_auxiliar2_error_forward_layer = NULL;
    if (d_auxiliar_error_forward_layer_pointers != NULL) { cudaFree(d_auxiliar_error_forward_layer_pointers); d_auxiliar_error_forward_layer_pointers = NULL; }
    if (d_auxiliar2_error_forward_layer_pointers != NULL) { cudaFree(d_auxiliar2_error_forward_layer_pointers); d_auxiliar2_error_forward_layer_pointers = NULL; }
    if (d_error_array_weight_matrix != NULL) { cudaFree(d_error_array_weight_matrix); d_error_array_weight_matrix = NULL; }
    if (d_error_array_bias_vector != NULL) { cudaFree(d_error_array_bias_vector); d_error_array_bias_vector = NULL; }
    if (hd_error_weight_matrices_pointers != NULL) { delete hd_error_weight_matrices_pointers;  hd_error_weight_matrices_pointers = NULL; }
    if (hd_error_bias_vectors_pointers != NULL) { delete hd_error_bias_vectors_pointers; hd_error_bias_vectors_pointers = NULL; }
    if (d_error_weight_matrices_pointers != NULL) { cudaFree(d_error_weight_matrices_pointers); d_error_weight_matrices_pointers = NULL; }
    if (d_error_bias_vectors_pointers != NULL) { cudaFree(d_error_bias_vectors_pointers); d_error_bias_vectors_pointers = NULL; }
}

void Layer::copyWeightBias(float* h_weight, float* h_bias) {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaMemcpy(d_array_weight_matrix, h_weight, input_size * size * number_networks * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array_bias_vector, h_bias, size * number_networks * sizeof(float), cudaMemcpyHostToDevice);
    }
}