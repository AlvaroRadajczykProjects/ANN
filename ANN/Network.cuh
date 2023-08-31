#include "Layer.cuh"

#ifndef NETWORK
#define NETWORK

class Network {
    private:
        int max_num_threads;

        int input_size;
        int output_size;
        int number_networks;
        int number_layers;
        Layer** layers;
        func2_t loss_function;
        func2_t derivative_loss_function;

        int max_train_number_examples;
        int max_validation_number_examples;
        int max_batch_size;
        int max_layer_size = 0;

        int current_train_number_examples;
        int current_validation_number_examples;
        int current_batch_size;

        bool check_errors_CUDA = false;

        cudaStream_t stream_principal;
        cudaStream_t stream_transferencia_output;

        float* h_pinned_input_train_matrix = NULL;
        float* h_pinned_output_train_matrix = NULL;
        float* d_pinned_input_train_matrix = NULL;
        float* d_pinned_output_train_matrix = NULL;

        float* h_pinned_input_validation_matrix = NULL;
        float* h_pinned_output_validation_matrix = NULL;
        float* d_pinned_input_validation_matrix = NULL;
        float* d_pinned_output_validation_matrix = NULL;

        //AUXILIAR_POINTERS
        //vector of 1's for repeating vector multiple times in matrix or sum all matrix cols in vector
        float* d_auxiliar_expand_reduce_matrix = NULL;
        //matrix of next four multiple of max_number_examples rows and output_size cols, when using more than one network, here the average of each network prediction is calculated
        //also is used for cost function summatory, perfectly fits in
        float* d_output_forward_multiple_nn_sum = NULL;
        float** d_output_forward_multiple_nn_sum_pointers = NULL;
        //matrix where changed-order output is copied and loss function is calculated, and also is stored backpropagated error of current layer
        float* d_auxiliar_matrix_loss_function_error_backprop = NULL;
        float* d_auxiliar2_matrix_loss_function_error_backprop = NULL;

        //also works as output pointers
        float** d_input_train_pointers = NULL;
        float** d_input_validation_pointers = NULL;

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
        void initForwardTrain(int max_train_examples, int max_validation_examples, int m_batch_size);

        void initWeightBiasValues();

        const void copyInputOutputTrain(int num_examples, float* input_data, float* output_data);
        const void copyInputOutputValidation(int num_examples, float* input_data, float* output_data);

        const void forward( int num_examples, float* input_data, float* output_pointer_dest);
        
        const void forwardTrain(int num_examples);
        const void forwardTrain(int num_examples, int batch_size, float** d_input_pointers);

        float* trainGetCostFunctionAndCalculateLossFunction(int num_examples, int offset_id);
        float* trainGetCostFunctionAndCalculateLossFunction(int num_examples, int batch_size, int* batch_ids);

        float* backwardPhase(int num_examples, int offset_id);
        float* backwardPhase(int num_examples, int batch_size, int* batch_ids);

        void applyVGradSGD(float lrate);

        float* validationGetCostFunctionAndCalculateLossFunction(int num_examples, int offset_id);
        float* validationGetCostFunctionAndCalculateLossFunction(int num_examples, int batch_size, int* batch_ids);

        void finalizeForwardBackward();

        //void initBackwardADAM();
        //void finalizeBackwardADAM();

        
        /*int getInputSize();
        int getOutputSize();
        
        void changeIsTraining(bool new_training);
        float* getWeightMatrix();
        float* getBiasVector();*/
        
};

#endif