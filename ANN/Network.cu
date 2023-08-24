#include "Network.cuh"

using namespace std;

Network::Network(int is, int nn, int nl, Layer** ls) {
	input_size = is;
	output_size = ls[nl - 1]->getSize();
	number_networks = nn;
	number_layers = nl;
	layers = ls;
	cublasCreate_v2(&handle);
	for (int i = 0; i < number_layers; i++) {
		max_layer_size = max(max_layer_size, layers[i]->getSize());
		layers[i]->setCublasHandle(&handle);
		layers[i]->setNumberNetworks(number_networks);
		if (i == 0) { layers[i]->setInputSize(input_size); }
		else { layers[i]->setInputSize(layers[i-1]->getSize()); }
		layers[i]->allocMemory();
	}
	cudaDeviceSynchronize();
}

Network::~Network() {
	for (int i = 0; i < number_layers; i++) {
		layers[i]->freeMemory();
		delete layers[i];
	}
	delete layers;
	cublasDestroy_v2(handle);
	cudaDeviceSynchronize();
}

void Network::showInfoAboutNetwork() {
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
	for (int i = 0; i < number_layers; i++) {
		layers[i]->showWeightBias();
	}
	printf("\n");
}

void Network::initForward(int max_num_input_examples_expected) {
	max_input_number_examples = max_num_input_examples_expected;
	cudaHostAlloc(&h_pinned_input_matrix, input_size * max_input_number_examples * sizeof(float), cudaHostAllocWriteCombined);
	cudaHostAlloc(&h_pinned_output_matrix, output_size * max_input_number_examples * sizeof(float), 0);
	d_pinned_output_offset = input_size * max_input_number_examples;
	cudaMalloc(&d_pinned_input_output_auxiliar_matrix, max_input_number_examples * (input_size + output_size) * sizeof(float));
	cudaDeviceSynchronize();
}

const void Network::forward(int num_examples, float* input_data, float* output_pointer_dest) {
	if (num_examples <= max_input_number_examples) {
		cudaMemcpyAsync(h_pinned_input_matrix, input_data, num_examples * input_size * sizeof(float), cudaMemcpyHostToHost);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix, h_pinned_input_matrix, num_examples * input_size * sizeof(float), cudaMemcpyHostToDevice);
	} else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForward");
	}
}

void Network::finalizeForward() {
	cudaFree(d_pinned_input_output_auxiliar_matrix);
	cudaFreeHost(h_pinned_input_matrix);
	cudaFreeHost(h_pinned_output_matrix);
	cudaDeviceSynchronize();
}

void Network::initForwardTrain(int max_num_input_examples_expected) {
	cublasCreate_v2(&handle);
	max_input_number_examples = max_num_input_examples_expected;
	cudaStreamCreate(&stream_principal);
	cudaStreamCreate(&stream_transferencia_output);
	cudaHostAlloc(&h_pinned_input_matrix, input_size * max_input_number_examples * sizeof(float), cudaHostAllocWriteCombined);
	cudaHostAlloc(&h_pinned_output_matrix, output_size * max_input_number_examples * sizeof(float), cudaHostAllocWriteCombined);
	d_pinned_output_offset = input_size * max_input_number_examples;
	cudaMalloc(&d_pinned_input_output_auxiliar_matrix, max_input_number_examples * max(number_networks * max_layer_size, input_size + output_size) * sizeof(float));
	cudaDeviceSynchronize();
}

const void Network::forwardTrain(int num_examples, float* input_data, float* output_data) {
	if (num_examples <= max_input_number_examples) {
		cudaMemcpyAsync(h_pinned_input_matrix, input_data, num_examples * input_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
		cudaMemcpyAsync(h_pinned_output_matrix, output_data, num_examples * output_size * sizeof(float), cudaMemcpyHostToHost, stream_transferencia_output);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix, h_pinned_input_matrix, num_examples * input_size * sizeof(float), cudaMemcpyHostToDevice, stream_principal);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix + d_pinned_output_offset, h_pinned_output_matrix, num_examples * output_size * sizeof(float), cudaMemcpyHostToDevice, stream_transferencia_output);
	}
	else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForward");
	}
}

void Network::finalizeForwardTrain() {
	cudaStreamDestroy(stream_principal);
	cudaStreamDestroy(stream_transferencia_output);
	cudaFree(d_pinned_input_output_auxiliar_matrix);
	cudaFreeHost(h_pinned_input_matrix);
	cudaFreeHost(h_pinned_output_matrix);
	cudaDeviceSynchronize();
}