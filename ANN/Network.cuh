#include "Layer.cuh"

#ifndef NETWORK
#define NETWORK

class Network {
    private:
        int max_num_threads;

        int input_size;
        int max_input_number_examples;
        int output_size;
        int number_networks;
        int number_layers;
        Layer** layers;
        int max_layer_size = 0;
        bool check_errors_CUDA = false;

        cudaStream_t stream_principal;
        cudaStream_t stream_transferencia_output;
        unsigned long long d_pinned_output_offset = 0;
        float* h_pinned_input_matrix;
        float* h_pinned_output_matrix;
        float* d_pinned_input_output_auxiliar_matrix;
        float* d_auxiliar_expand_reduce_matrix;
        float* d_output_forward_multiple_nn_sum;
        float** d_input_pointers = 0;

        cublasHandle_t handle;

        /*func2_t loss_function = NULL;
        */

    public:
        Network( int is, int nn, int nl, Layer** ls );
        ~Network();

        void showInfoAboutNetwork();
        void showWeightsBiasesLayers();
        void showAuxiliarExpandReduceMatrices();
        void showForwardMatrices();

        void initForward( int max_num_input_examples_expected );
        const void forward( int num_examples, float* input_data, float* output_pointer_dest);
        const void forwardTrain(int num_examples, float* input_data, float* output_data);
        void finalizeForward();

        //void initBackwardADAM();
        //void finalizeBackwardADAM();

        /*int getInputSize();
        int getOutputSize();
        int getNumberNetwors();
        void changeIsTraining(bool new_training);
        float* getWeightMatrix();
        float* getBiasVector();*/
        
};

#endif