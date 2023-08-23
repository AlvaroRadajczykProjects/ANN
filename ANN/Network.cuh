#include "Layer.cuh"

#ifndef NETWORK
#define NETWORK

class Network {
    private:
        int input_size;
        int max_input_examples;
        int output_size;
        int number_networks;
        int number_layers;
        Layer** layers;
        int max_layer_size = 0;

        /*func2_t loss_function = NULL;

        unsigned long long d_pinned_output_offset = 0;

        float* h_pinned_input_matrix;
        float* h_pinned_output_matrix;

        float* d_pinned_input_output_auxiliar_matrix;
        */

    public:
        Network( int is, int nn, int nl, Layer** ls );

        void showInfoAboutNetwork();

        /*int getInputSize();
        int getOutputSize();
        int getNumberNetwors();
        void changeIsTraining(bool new_training);
        float* getWeightMatrix();
        float* getBiasVector();*/
        
};

#endif