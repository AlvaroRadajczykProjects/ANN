#include "Network.cuh"

using namespace std;

Network::Network(int is, int nn, int nl, Layer** ls) {
	input_size = is;
	output_size = ls[nl - 1]->getSize();
	number_networks = nn;
	number_layers = nl;
	layers = ls;
	for (int i = 0; i < number_layers; i++) {
		max_layer_size = max(max_layer_size, layers[i]->getSize());
	}
	showInfoAboutNetwork();
}

void Network::showInfoAboutNetwork() {
	printf("\nInput size (number of each example attributes): %d", input_size);
	//printf("\nMax input examples (this network can forward training and/or predicting): %d", input_size);
	printf("\nOutput size (number of each prediction): %d", output_size);
	printf("\nNumber of networks (multiple networks can be trained for ensemble averaging with multiple similar neural networks in one device): %d", number_networks);
	printf("\nNumber of layers (all networks are similar, same shape, different initialization values): %d", number_layers);
	printf("\nMax layer size: %d", max_layer_size);
	printf("\n");
}