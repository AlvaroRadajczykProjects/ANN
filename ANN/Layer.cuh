#include "CUDAKernels.cuh"
#include "PrintUtils.h"

#ifndef LAYER
#define LAYER

class Layer {
    private:
        int input_size = 0;
        int size = 0;
        int number_networks = 0;
        int number_input_examples = 0;
        bool is_training = false;

        func_t activation_function = NULL;
        func_t activation_derivative_function = NULL;
        //func2_t each_output_sum = NULL;
        //func2_t activation_function2 = NULL;
        //func2_t activation_derivative_function2 = NULL;

        float* d_array_weight_matrix = NULL;
        float* d_array_bias_vector = NULL;
        float** d_weight_matrices = NULL;
        float** d_bias_vectors = NULL;

        //sólo hace falta al, este se puede deshacer vuelta a zl haciendo las operaciones opuestas al revés en la función de la derivada
        float* d_forward = NULL;

        float* d_error_weight_matrix = NULL;
        float* d_error_bias_vector = NULL;

        float* d_auxiliar_transpose_matrix = NULL;
        //será la matriz de device de tamaño max(nelems_entrada+nelems_salida, nelems_mayor_capa_salida)
        float* d_auxiliar_error_forward_layer = NULL;

        float* d_weight_matrix_momentum = NULL;
        float* d_bias_vector_momentum = NULL;
        float* d_weight_matrix_velocity = NULL;
        float* d_bias_vector_velocity = NULL;
        
    public:
        //dev_act_func and dev_act_der_func are __device__ float func(), need to be casted to (const void*)
        Layer( int sz, func_t dev_act_func, func_t dev_act_der_func );
        //~Layer();

        int getSize();
        /*int getInputSize();
        
        int getNumberNetwors();
        float* getHostWeightMatrix();
        float* getHostBiasVector();
        float* getDeviceForward();

        void setInputSize(int sz);
        void setNumberNetwors(int nn);
        void setAuxiliarTransposeMatrix(float* d_aux_mtr);
        void setIsTraining(bool new_training);

        void forward(float* d_input_values);
        void forward(Layer* previous_layer);

        float obtainCostFunction();

        void applyLossFunction();*/
};

#endif