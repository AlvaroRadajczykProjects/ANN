#include "Network.cuh"

using namespace std;

Network::Network(int is, int nn, int nl, Layer** ls) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	max_num_threads = deviceProp.maxThreadsPerBlock;

	input_size = is;
	output_size = ls[nl - 1]->getSize();
	number_networks = nn;
	number_layers = nl;
	layers = ls;
	layers[0]->setIsFirstLayer(true);
	cublasCreate_v2(&handle);
	for (int i = 0; i < number_layers; i++) {
		max_layer_size = max(max_layer_size, layers[i]->getSize());
		layers[i]->setCublasHandle(&handle);
		layers[i]->setNumberNetworks(number_networks);
		layers[i]->setIsTraining(false);
		if (i == 0) { layers[i]->setInputSize(input_size); }
		else { layers[i]->setInputSize(layers[i-1]->getSize()); }
		layers[i]->setMaxNumThreads(max_num_threads);
		layers[i]->allocWeightMatricesMemory();
	}
	cudaDeviceSynchronize();
}

Network::~Network() {
	for (int i = 0; i < number_layers; i++) {
		layers[i]->freeWeightMatricesMemory();
		delete layers[i];
	}
	delete layers;
	cublasDestroy_v2(handle);
	cudaDeviceSynchronize();
}

void Network::showInfoAboutNetwork() {
	printf("\n");
	printf("\nINFO ABOUT THE NETWORK");
	printf("\n======================");
	printf("\nInput size (number of each example attributes): %d", input_size);
	//printf("\nMax input examples (this network can forward training and/or predicting): %d", input_size);
	printf("\nOutput size (number of each prediction): %d", output_size);
	printf("\nNumber of networks (multiple networks can be trained for ensemble averaging with multiple similar neural networks in one device): %d", number_networks);
	printf("\nNumber of layers (all networks are similar, same shape, different initialization values): %d", number_layers);
	printf("\nMax layer size: %d", max_layer_size);
	printf("\nLayers dimensions:");
	for (int i = 0; i < number_layers; i++) {
		printf("\n\tLayer %d:", i);
		layers[i]->showInfo();
	}
	printf("\n");
}

void Network::showWeightsBiasesLayers() {
	printf("\n");
	printf("\nWEIGHTS AND BIASES");
	printf("\n==================");
	for (int i = 0; i < number_layers; i++) {
		printf("\nLayer %d:",i);
		layers[i]->showWeightBias();
	}
	printf("\n");
}

void Network::showAuxiliarExpandReduceMatrices() {
	printf("\n");
	printf("\nAUXILIAR EXPAND AND REDUCE VECTORS (is only one, but check all networks match the same)");
	printf("\n==================================------------------------------------------------------");
	printf("\n");
	for (int i = 0; i < number_layers; i++) {
		printf("\nLayer %d:", i);
		layers[i]->showAuxiliarExpandReduce();
	}
	printf("\n");
}

void Network::showForwardMatrices() {
	printf("\n");
	printf("\nFORWARD MATRICES");
	printf("\n================");
	for (int i = 0; i < number_layers; i++) {
		printf("\nLayer %d:", i);
		layers[i]->showForward();
	}
	printf("\n");
}

void Network::initForward(int max_num_input_examples_expected) {
	max_input_number_examples = max_num_input_examples_expected;
	d_pinned_output_offset = input_size * max_input_number_examples;
	cudaStreamCreate(&stream_principal);
	cudaStreamCreate(&stream_transferencia_output);
	cublasSetStream_v2(handle, stream_principal);
	cudaHostAlloc(&h_pinned_input_matrix, input_size * max_input_number_examples * sizeof(float), cudaHostAllocWriteCombined);
	cudaHostAlloc(&h_pinned_output_matrix, output_size * max_input_number_examples * sizeof(float), cudaHostAllocWriteCombined);
	cudaMalloc(&d_pinned_input_output_auxiliar_matrix, max_input_number_examples * ( input_size + output_size) * sizeof(float));

	float** hd_input_pointers = new float* [number_networks];
	for (int i = 0; i < number_networks; i++) { hd_input_pointers[i] = d_pinned_input_output_auxiliar_matrix + 0; }
	cudaMalloc(&d_input_pointers, number_networks*sizeof(float*));
	cudaMemcpy(d_input_pointers, hd_input_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
	delete hd_input_pointers;
	
	cudaMalloc(&d_auxiliar_expand_reduce_matrix, max_num_input_examples_expected * sizeof(float));
	float* h_auxiliar_expand_reduce_matrix = new float[max_num_input_examples_expected];
	for (int i = 0; i < max_num_input_examples_expected; i++) { h_auxiliar_expand_reduce_matrix[i] = 1.0f; }
	cudaMemcpy(d_auxiliar_expand_reduce_matrix, h_auxiliar_expand_reduce_matrix, max_num_input_examples_expected * sizeof(float), cudaMemcpyHostToDevice);
	delete h_auxiliar_expand_reduce_matrix;
	for (int i = 0; i < number_layers; i++) {
		layers[i]->setNumberInputExamples(max_input_number_examples);
		layers[i]->setAuxiliarExpandReduceMatrix(d_auxiliar_expand_reduce_matrix);
		layers[i]->allocForwardMemory();
	}
	cudaDeviceSynchronize();
}

const void Network::forward(int num_examples, float* input_data, float* output_pointer_dest) {
	if (num_examples <= max_input_number_examples) {
		cudaMemcpyAsync(h_pinned_input_matrix, input_data, num_examples * input_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix, h_pinned_input_matrix, num_examples * input_size * sizeof(float), cudaMemcpyHostToDevice, stream_principal);
		layers[0]->forward(stream_principal, d_input_pointers);
		for (int i = 1; i < number_layers; i++) {
			layers[i]->forward(stream_principal, layers[i-1]);
		}
		cudaStreamSynchronize(stream_principal);
	} else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForward");
	}
}

const void Network::forwardTrain(int num_examples, float* input_data, float* output_data) {
	if (num_examples <= max_input_number_examples) {
		cudaMemcpyAsync(h_pinned_input_matrix, input_data, num_examples * input_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
		cudaMemcpyAsync(h_pinned_output_matrix, output_data, num_examples * output_size * sizeof(float), cudaMemcpyHostToHost, stream_transferencia_output);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix, h_pinned_input_matrix, num_examples * input_size * sizeof(float), cudaMemcpyHostToDevice, stream_principal);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix + d_pinned_output_offset, h_pinned_output_matrix, num_examples * output_size * sizeof(float), cudaMemcpyHostToDevice, stream_transferencia_output);
		//para hacer el backward, esperaré a que ambas transferencias hayan terminado
		cudaStreamSynchronize(stream_principal);
		cudaStreamSynchronize(stream_transferencia_output);
	}
	else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForward");
	}
}

void Network::finalizeForward() {
	cublasSetStream_v2(handle, 0);
	cudaStreamDestroy(stream_principal);
	cudaStreamDestroy(stream_transferencia_output);
	cudaFree(d_pinned_input_output_auxiliar_matrix);
	cudaFree(d_input_pointers);
	cudaFreeHost(h_pinned_input_matrix);
	cudaFreeHost(h_pinned_output_matrix);
	
	for (int i = 0; i < number_layers; i++) {
		layers[i]->setNumberInputExamples(0);
		layers[i]->setAuxiliarExpandReduceMatrix(NULL);
		layers[i]->freeForwardMemory();
	}
	cudaFree(d_auxiliar_expand_reduce_matrix);
	cudaDeviceSynchronize();
}