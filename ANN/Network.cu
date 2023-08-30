#include "Network.cuh"

using namespace std;

Network::Network(int is, int nn, int nl, Layer** ls, func2_t ls_fn, func2_t dls_fn) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	max_num_threads = deviceProp.maxThreadsPerBlock;

	loss_function = ls_fn;
	derivative_loss_function = dls_fn;
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

void Network::showErrorWeightsBiasesLayers() {
	printf("\n");
	printf("\nERROR WEIGHTS AND BIASES");
	printf("\n========================");
	for (int i = 0; i < number_layers; i++) {
		printf("\nLayer %d:", i);
		layers[i]->showErrorWeightBias();
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

int Network::getNumberNetwors() {
	return number_networks;
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
	
	int tam = nextFourMultiple(max(max(max_batch_size, number_networks), output_size));
	cudaMalloc(&d_auxiliar_expand_reduce_matrix, tam * sizeof(float));
	float* h_auxiliar_expand_reduce_matrix = new float[tam];
	for (int i = 0; i < tam; i++) { h_auxiliar_expand_reduce_matrix[i] = 1.0f; }
	cudaMemcpy(d_auxiliar_expand_reduce_matrix, h_auxiliar_expand_reduce_matrix, tam * sizeof(float), cudaMemcpyHostToDevice);
	delete h_auxiliar_expand_reduce_matrix;
	for (int i = 0; i < number_layers; i++) {
		layers[i]->setNumberInputExamples(max_input_number_examples);
		layers[i]->setAuxiliarExpandReduceMatrix(d_auxiliar_expand_reduce_matrix);
		layers[i]->allocForwardMemory();
	}

	cudaMalloc(&d_output_forward_multiple_nn_sum, nextFourMultiple(number_networks * max_input_number_examples * output_size) * sizeof(float));

	//Cublas warmup
	productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, layers[number_layers - 1]->getDeviceForward(), d_output_forward_multiple_nn_sum, 1, number_networks, output_size);

	cudaDeviceSynchronize();
}

void Network::initForwardTrain(int m_num_examples, int m_batch_size) {
	max_batch_size = m_batch_size;
	max_input_number_examples = m_num_examples;
	d_pinned_output_offset = input_size * max_input_number_examples;
	cudaStreamCreate(&stream_principal);
	cudaStreamCreate(&stream_transferencia_output);
	cublasSetStream_v2(handle, stream_principal);
	cudaHostAlloc(&h_pinned_input_matrix, input_size * max_input_number_examples * sizeof(float), cudaHostAllocWriteCombined);
	cudaHostAlloc(&h_pinned_output_matrix, output_size * max_input_number_examples * sizeof(float), cudaHostAllocWriteCombined);
	cudaMalloc(&d_pinned_input_output_auxiliar_matrix, ( d_pinned_output_offset + nextFourMultiple(max_input_number_examples * output_size) ) * sizeof(float));

	float** hd_input_pointers = new float* [number_networks];
	for (int i = 0; i < number_networks; i++) { hd_input_pointers[i] = d_pinned_input_output_auxiliar_matrix + 0; }
	cudaMalloc(&d_input_pointers, number_networks * sizeof(float*));
	cudaMemcpy(d_input_pointers, hd_input_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
	delete hd_input_pointers;

	int tam = nextFourMultiple( max_batch_size * number_networks * output_size );
	cudaMalloc(&d_auxiliar_expand_reduce_matrix, tam * sizeof(float));
	float* h_auxiliar_expand_reduce_matrix = new float[tam];
	for (int i = 0; i < tam; i++) { h_auxiliar_expand_reduce_matrix[i] = 1.0f; }
	cudaMemcpy(d_auxiliar_expand_reduce_matrix, h_auxiliar_expand_reduce_matrix, tam * sizeof(float), cudaMemcpyHostToDevice);
	delete h_auxiliar_expand_reduce_matrix;

	//max_input_number_examples instead of max_batch_size * number_networks?
	cudaMalloc(&d_auxiliar_matrix_loss_function_error_backprop, nextFourMultiple(max_batch_size * number_networks * max_layer_size) * sizeof(float));
	cudaMalloc(&d_auxiliar2_matrix_loss_function_error_backprop, nextFourMultiple(max_batch_size * number_networks * max_layer_size) * sizeof(float));

	for (int i = 0; i < number_layers; i++) {
		layers[i]->setNumberInputExamples(max_batch_size);
		layers[i]->setAuxiliarExpandReduceMatrix(d_auxiliar_expand_reduce_matrix);
		layers[i]->allocForwardMemory();
		layers[i]->allocBackwardMemory(m_batch_size, d_auxiliar_matrix_loss_function_error_backprop, d_auxiliar2_matrix_loss_function_error_backprop);
		layers[i]->setIsTraining(true);
	}

	cudaMalloc(&d_output_forward_multiple_nn_sum, nextFourMultiple(number_networks * max_batch_size * output_size) * sizeof(float));
	float** hd_output_forward_multiple_nn_sum_pointers = new float* [number_networks];
	cudaMalloc(&d_output_forward_multiple_nn_sum_pointers, number_networks * sizeof(float*));
	for (int i = 0; i < number_networks; i++) { hd_output_forward_multiple_nn_sum_pointers[i] = d_output_forward_multiple_nn_sum + i*output_size; }
	cudaMemcpy(d_output_forward_multiple_nn_sum_pointers, hd_output_forward_multiple_nn_sum_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
	delete hd_output_forward_multiple_nn_sum_pointers;

	int first_max = max(max_layer_size, input_size);
	int second_max = 0;
	for (int i = 0; i < number_layers; i++) {
		if (max(second_max, layers[i]->getSize()) < first_max) { second_max = max(second_max, layers[i]->getSize()); }
	}

	//Cublas warmup
	productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, layers[number_layers - 1]->getDeviceForward(), d_output_forward_multiple_nn_sum, 1, number_networks, output_size);

	cudaDeviceSynchronize();
}

void Network::initWeightBiasValues() {
	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MT19937);
	curandSetGeneratorOrdering(curandGenerator, CURAND_ORDERING_PSEUDO_BEST);
	cudaDeviceSynchronize();
	for (int i = 0; i < number_layers; i++) {
		layers[i]->initWeightBiasValues(curandGenerator);
	}
	cudaDeviceSynchronize();
	curandDestroyGenerator(curandGenerator);
}

const void Network::copyInputOutputTrain(int num_examples, float* input_data, float* output_data) {
	if (num_examples <= max_input_number_examples) {
		cudaMemcpyAsync(h_pinned_input_matrix, input_data, num_examples * input_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
		cudaMemcpyAsync(h_pinned_output_matrix, output_data, num_examples * output_size * sizeof(float), cudaMemcpyHostToHost, stream_transferencia_output);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix, h_pinned_input_matrix, num_examples * input_size * sizeof(float), cudaMemcpyHostToDevice, stream_principal);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix + d_pinned_output_offset, h_pinned_output_matrix, num_examples * output_size * sizeof(float), cudaMemcpyHostToDevice, stream_transferencia_output);
		//cudaStreamSynchronize(stream_principal);
		//cudaStreamSynchronize(stream_transferencia_output);
		//para hacer el backward, esperaré a que ambas transferencias hayan terminado
		/*
		//input y output se copian bien, deberían de llegar a tiempo sin necesidad de sincronización
		float* inputarr = new float[input_size * num_examples];
		float* outputarr = new float[output_size * num_examples];
		cudaMemcpy(inputarr, d_pinned_input_output_auxiliar_matrix, input_size * num_examples * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(outputarr, d_pinned_input_output_auxiliar_matrix + d_pinned_output_offset, output_size * num_examples * sizeof(float), cudaMemcpyDeviceToHost);
		imprimirMatrizPorPantalla("input", inputarr, num_examples, input_size);
		imprimirMatrizPorPantalla("output", outputarr, num_examples, output_size);
		*/
	}
	else {
		printf("\nCannot copy input and output, more examples than max number of examples defined in initForward");
	}
}

const void Network::forward(int num_examples, float* input_data, float* output_pointer_dest) {
	if (num_examples <= max_input_number_examples) {
		cudaMemcpyAsync(h_pinned_input_matrix, input_data, num_examples * input_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
		cudaMemcpyAsync(d_pinned_input_output_auxiliar_matrix, h_pinned_input_matrix, num_examples * input_size * sizeof(float), cudaMemcpyHostToDevice, stream_principal);
		layers[0]->forward(stream_principal, d_input_pointers, num_examples);
		for (int i = 1; i < number_layers; i++) {
			layers[i]->forward(stream_principal, layers[i-1], num_examples);
		}

		if (number_networks == 1) {
			cudaMemcpyAsync(h_pinned_output_matrix, layers[number_layers-1]->getDeviceForward(), num_examples * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_principal);
			cudaMemcpyAsync(output_pointer_dest, h_pinned_output_matrix, num_examples * output_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
		} else {
			if (max_input_number_examples == 1) {
				productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, layers[number_layers - 1]->getDeviceForward(), d_output_forward_multiple_nn_sum, 1, number_networks, output_size);
				
				managedMultiplyAllElementsByConstant(stream_principal, max_num_threads, nextFourMultiple(output_size), d_output_forward_multiple_nn_sum, 1 / (float)number_networks);
				//multiplyAllElementsByConstantVectorial << < (int)ceil(nextFourMultiple(num_examples * output_size) /(float)(max_num_threads*4)), min(max_num_threads, nextFourMultiple(num_examples * output_size) / 4), 0, stream_principal >> > (d_output_forward_multiple_nn_sum, 1 / (float)number_networks);
				cudaMemcpyAsync(h_pinned_output_matrix, d_output_forward_multiple_nn_sum, num_examples * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_principal);
				cudaMemcpyAsync(output_pointer_dest, h_pinned_output_matrix, num_examples * output_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
			} else {
				//habrá que hacer el sumatorio de todas las matrices al de todas las redes, y multiplicarles 1/numero_redes
			}
		}
		cudaStreamSynchronize(stream_principal);
	} else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForward");
	}
}

const void Network::forwardTrain(int num_examples) {
	layers[0]->forward(stream_principal, d_input_pointers, num_examples);
	for (int i = 1; i < number_layers; i++) {
		layers[i]->forward(stream_principal, layers[i - 1], num_examples);
	}
	cudaStreamSynchronize(stream_principal);
	cudaStreamSynchronize(stream_transferencia_output);
}

//first batch_id = 0
const void Network::forwardTrain(int num_examples, int batch_size, float** d_input_pointers) {
	layers[0]->forward(stream_principal, d_input_pointers, batch_size);
	for (int i = 1; i < number_layers; i++) {
		layers[i]->forward(stream_principal, layers[i - 1], batch_size);
	}
	cudaStreamSynchronize(stream_principal);
	cudaStreamSynchronize(stream_transferencia_output);
}

float* Network::trainGetCostFunctionAndCalculateLossFunction(int num_examples) {
	int* pos = new int[number_networks];
	for (int i = 0; i < number_networks; i++) { pos[i] = 0; }
	return trainGetCostFunctionAndCalculateLossFunction(num_examples, num_examples, pos);
}

//first batch_id = 0
float* Network::trainGetCostFunctionAndCalculateLossFunction(int num_examples, int batch_size, int* batch_ids) {
	if (batch_size <= max_input_number_examples) {
		if (num_examples % batch_size == 0) {
			int num_elems_batch = batch_size * input_size;
			float** ptrs = new float* [number_networks];
			for (int i = 0; i < number_networks; i++) { ptrs[i] = d_pinned_input_output_auxiliar_matrix + (batch_ids[i] * num_elems_batch); }
			cudaMemcpy(d_input_pointers, ptrs, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
			forwardTrain(num_examples, batch_size, d_input_pointers);
			delete ptrs;

			num_elems_batch = batch_size * output_size;
			
			/*
			float* matriz_Cost2 = new float[num_elems_batch];
			for (int i = 0; i < number_networks; i++) {
				cudaMemcpy(matriz_Cost, layers[number_layers - 1]->getDeviceForward() + (i * num_elems_batch), num_elems_batch * sizeof(float), cudaMemcpyDeviceToHost);
				imprimirMatrizPorPantalla("XD:", matriz_Cost, batch_size, output_size);
				cudaMemcpy(matriz_Cost, d_pinned_input_output_auxiliar_matrix + d_pinned_output_offset + (batch_ids[i] * num_elems_batch), num_elems_batch * sizeof(float), cudaMemcpyDeviceToHost);
				imprimirMatrizPorPantalla("XD2:", matriz_Cost, batch_size, output_size);
			}
			delete matriz_Cost2;
			*/

			//apply cost function
			for (int i = 0; i < number_networks; i++) {
				managedApplyLossFunction(stream_principal, max_num_threads, num_elems_batch,
					layers[number_layers - 1]->getDeviceForward() + (i * num_elems_batch),
					d_pinned_input_output_auxiliar_matrix + d_pinned_output_offset + (batch_ids[i] * num_elems_batch),
					d_auxiliar_matrix_loss_function_error_backprop + (i * num_elems_batch),
					loss_function
				);
			}

			//obtain cost function
			float* cost_function_result = new float[number_networks];
			float* d_res = 0;
			cudaMalloc(&d_res,number_networks * sizeof(float));
			productoMatricesBatchDevice(handle, layers[number_layers-1]->getAuxiliarExpandReduceMatrixPointers(), layers[number_layers - 1]->getDeviceAuxiliarErrorForwardLayerPointers(), d_output_forward_multiple_nn_sum_pointers, 1, batch_size, output_size, number_networks);
			
			managedMultiplyAllElementsByConstant(stream_principal, max_num_threads, output_size * number_networks, d_output_forward_multiple_nn_sum, 1 / (float)(batch_size));

			/*multiplyAllElementsByConstantVectorial << < (int)ceil(output_size * number_networks / 4), min(max_num_threads, (int)(output_size * number_networks / 4)), 0, stream_principal >> > (d_output_forward_multiple_nn_sum, 1 / (float)(batch_size));
			if ((output_size * number_networks) % 4 != 0) {
				multiplyAllElementsByConstantScalar << < 1, (output_size * number_networks) % 4, 0, stream_principal >> > (d_output_forward_multiple_nn_sum + (((output_size * number_networks) / 4) * 4), 1 / (float)(batch_size));
			}
			*/
			productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, d_output_forward_multiple_nn_sum, d_res, 1, output_size, number_networks);
			cudaMemcpy(cost_function_result, d_res, number_networks*sizeof(float), cudaMemcpyDeviceToHost);
			for(int i = 0; i < number_networks;i++){ cost_function_result[i] = cost_function_result[i] / (float) output_size; }
			/*for (int i = 0; i < number_networks; i++) {
				productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, d_auxiliar_matrix_loss_function_error_backprop + i*batch_size, d_output_forward_multiple_nn_sum, 1, batch_size, output_size);
				multiplyAllElementsByConstantVectorial << < (int)ceil(output_size / 4), min(max_num_threads, (int)(output_size / 4)), 0, stream_principal >> > (d_output_forward_multiple_nn_sum, 1 / (float)(batch_size));
				if (output_size % 4 != 0) {
					multiplyAllElementsByConstantScalar << < 1, output_size % 4, 0, stream_principal >> > (d_output_forward_multiple_nn_sum + ((output_size / 4) * 4), 1 / (float)(batch_size));
				}

				productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, d_output_forward_multiple_nn_sum, d_res, 1, output_size, 1);
				cudaMemcpy(&cost_function_result[i], d_res, sizeof(float), cudaMemcpyDeviceToHost);
				cost_function_result[i] = cost_function_result[i] / (float)output_size;
			}*/
			cudaFree(d_res);

			/*
			float* matriz_Cost = new float[num_elems_batch * number_networks];
			cudaMemcpy(matriz_Cost, d_auxiliar_matrix_loss_function_error_backprop, num_elems_batch * number_networks * sizeof(float), cudaMemcpyDeviceToHost);
			imprimirMatrizPorPantalla("Error de coste:", matriz_Cost, batch_size* number_networks, output_size);
			delete matriz_Cost;
			*/

			/*
			float* mirarCost = new float[number_networks*output_size];
			cudaMemcpy(mirarCost, d_output_forward_multiple_nn_sum, number_networks * output_size * sizeof(float), cudaMemcpyDeviceToHost);
			imprimirMatrizPorPantalla("Error de coste sum:", mirarCost, number_networks, output_size);
			delete mirarCost;
			*/

			//apply loss function
			for (int i = 0; i < number_networks; i++) {
				managedApplyLossFunction(stream_principal, max_num_threads, num_elems_batch,
					layers[number_layers - 1]->getDeviceForward() + (i * num_elems_batch),
					d_pinned_input_output_auxiliar_matrix + d_pinned_output_offset + (batch_ids[i] * num_elems_batch),
					d_auxiliar_matrix_loss_function_error_backprop + (i * num_elems_batch),
					derivative_loss_function
				);
			}

			/*
			float* matriz_Cost = new float[num_elems_batch * number_networks];
			cudaMemcpy(matriz_Cost, d_auxiliar_matrix_loss_function_error_backprop, num_elems_batch * number_networks * sizeof(float), cudaMemcpyDeviceToHost);
			imprimirMatrizPorPantalla("Loss function applied:", matriz_Cost, batch_size* number_networks, output_size);
			delete matriz_Cost;
			*/

			return cost_function_result;
		} else {
			printf("\nwhen batch forwardTrain, num_examples % batch_size must be 0");
		}
	} else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForward");
	}
	return NULL;
}

float* Network::backwardPhase(int num_examples, int batch_size, int* batch_ids) {
	if (batch_size <= max_input_number_examples) {
		if (num_examples % batch_size == 0) {
			float* cost_function = trainGetCostFunctionAndCalculateLossFunction(num_examples, batch_size, batch_ids);
			for (int i = number_layers - 1; i > 0; i--) {
				layers[i]->backward(stream_principal, layers[i - 1], batch_size);
			}
			layers[0]->backward(stream_principal, d_input_pointers, batch_size);
			return cost_function;
		} else {
			printf("\nwhen batch forwardTrain, num_examples % batch_size must be 0");
		}
	} else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForward");
	}
	return NULL;
}

void Network::applyVGradSGD(float lrate) {
	for (int i = 0; i < number_layers; i++) {
		layers[i]->applyGradientSGD(stream_principal, lrate);
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
		layers[i]->freeBackwardMemory();
		layers[i]->setIsTraining(false);
	}
	cudaFree(d_auxiliar_expand_reduce_matrix);

	cudaFree(d_output_forward_multiple_nn_sum);
	if (d_output_forward_multiple_nn_sum_pointers != NULL) { cudaFree(d_output_forward_multiple_nn_sum_pointers);  d_output_forward_multiple_nn_sum_pointers = NULL; }

	if (d_auxiliar_matrix_loss_function_error_backprop != NULL) { cudaFree(d_auxiliar_matrix_loss_function_error_backprop); d_auxiliar_matrix_loss_function_error_backprop = NULL; }
	if (d_auxiliar2_matrix_loss_function_error_backprop != NULL) { cudaFree(d_auxiliar2_matrix_loss_function_error_backprop); d_auxiliar2_matrix_loss_function_error_backprop = NULL; }

	cudaDeviceSynchronize();
	max_batch_size = 0;
	max_input_number_examples = 0;
}