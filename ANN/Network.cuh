#include "Layer.cuh"

#ifndef NETWORK
#define NETWORK

class Network {
    private:
        int max_num_threads;

        int input_size;
        int max_input_number_examples;
        int max_batch_size;
        int output_size;
        int number_networks;
        int number_layers;
        Layer** layers;
        func2_t loss_function;
        func2_t derivative_loss_function;
        int max_layer_size = 0;
        bool check_errors_CUDA = false;

        cudaStream_t stream_principal;
        cudaStream_t stream_transferencia_output;
        unsigned long long d_pinned_output_offset = 0;
        float* h_pinned_input_matrix;
        float* h_pinned_output_matrix;
        float* d_pinned_input_output_auxiliar_matrix;

        //AUXILIAR_POINTERS
        //vector of 1's for repeating vector multiple times in matrix or sum all matrix cols in vector
        float* d_auxiliar_expand_reduce_matrix;
        //matrix of next four multiple of max_number_examples rows and output_size cols, when using more than one network, here the average of each network prediction is calculated
        //also is used for cost function summatory, perfectly fits in
        float* d_output_forward_multiple_nn_sum;
        float** d_output_forward_multiple_nn_sum_pointers = NULL;
        //matrix where changed-order output is copied and loss function is calculated, and also is stored backpropagated error of current layer
        float* d_auxiliar_matrix_loss_function_error_backprop = NULL;
        float* d_auxiliar2_matrix_loss_function_error_backprop = NULL;

        //also works as output pointers
        float** d_input_pointers = 0;

        cublasHandle_t handle;

        /*func2_t loss_function = NULL;
        */

    public:
        Network( int is, int nn, int nl, Layer** ls, func2_t ls_fn, func2_t dls_fn);
        ~Network();

        void showInfoAboutNetwork();
        void showWeightsBiasesLayers();
        void showErrorWeightsBiasesLayers();
        void showAuxiliarExpandReduceMatrices();
        void showForwardMatrices();

        int getNumberNetwors();

        void initForward( int max_num_input_examples_expected );
        void initForwardTrain(int num_examples, int max_batch_size);

        void initWeightBiasValues();

        const void copyInputOutputTrain(int num_examples, float* input_data, float* output_data);

        const void forward( int num_examples, float* input_data, float* output_pointer_dest);
        
        const void forwardTrain(int num_examples);
        const void forwardTrain(int num_examples, int batch_size, float** d_input_pointers);

        float* trainGetCostFunctionAndCalculateLossFunction(int num_examples);
        float* trainGetCostFunctionAndCalculateLossFunction(int num_examples, int batch_size, int* batch_ids);

        float* backwardPhase(int num_examples, int batch_size, int* batch_ids);

        void applyVGradSGD(float lrate);

        void finalizeForward();

        //void initBackwardADAM();
        //void finalizeBackwardADAM();

        
        /*int getInputSize();
        int getOutputSize();
        
        void changeIsTraining(bool new_training);
        float* getWeightMatrix();
        float* getBiasVector();*/
        
};

#endif