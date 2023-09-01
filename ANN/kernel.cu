#include <stdio.h>
#include <chrono>

#include "Network.cuh"

#define N 20

extern __device__ func_t d_ELU;
extern __device__ func_t d_dELU;
extern __device__ func_t d_Linear;
extern __device__ func_t d_dLinear;
extern __device__ func2_t d_MSE;
extern __device__ func2_t d_dMSE;

func_t getDeviceSymbolInGlobalMemory(func_t d_arrfunc) {
    func_t h_arrfunc;
    cudaMemcpy(&h_arrfunc, d_arrfunc, sizeof(func_t), cudaMemcpyDeviceToHost);
    return h_arrfunc;
}

func2_t getDeviceSymbolInGlobalMemory(func2_t d_func) {
    func2_t h_func;
    cudaMemcpy(&h_func, d_func, sizeof(func2_t), cudaMemcpyDeviceToHost);
    return h_func;
}

func_t d_func = 0;
func2_t d_func2 = 0;
func3_t d_func3 = 0;

int main() {

    srand(time(NULL));
    
    cudaMalloc(&d_func, sizeof(func_t));

    cudaGetSymbolAddress((void**)&d_func, d_ELU);
    func_t ELU = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_dELU);
    func_t dELU = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_Linear);
    func_t Linear = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_dLinear);
    func_t dLinear = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func2, d_MSE);
    func2_t MSE = getDeviceSymbolInGlobalMemory(d_func2);

    cudaGetSymbolAddress((void**)&d_func2, d_dMSE);
    func2_t dMSE = getDeviceSymbolInGlobalMemory(d_func2);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int m = 16;

    ///*
    Network* n = new Network(m, 1, 3, new Layer * [3] {
        new Layer(m, ELU, dELU),
        new Layer(m, ELU, dELU),
        new Layer(m, Linear, dLinear),
    }, MSE, dMSE);

    n->initWeightBiasValues();

    n->showWeightsBiasesLayers();

    float* input = new float[m];
    float* output = new float[m];

    n->initForwardTrain(1, 1, 1);

    n->copyInputOutputTrain(1, input, output);
    n->copyInputOutputValidation(1, input, output);

    //n->trainAllExamplesMaxBatchSGD(20000, 500, 0.00000001, 0.0000001, 6, 0.01);

    //n->trainGetCostFunctionAndCalculateLossFunction(1, 0);
    //n->showForwardMatrices();

    n->finalizeForwardBackward();

    delete n;
    //*/

    /*
    Network* n = new Network(2, 5, 3, new Layer * [3] {
        new Layer(10, ELU, dELU),
            new Layer(10, ELU, dELU),
            new Layer(1, Linear, dLinear),
    }, MSE, dMSE);

    n->initWeightBiasValues();

    float* input = new float[4 * 2] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* output = new float[4 * 1] { 1, 0, 0, 1 };

    float* input_val = new float[4 * 2] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* output_val = new float[4 * 1] { 1, 0, 0, 1 };

    n->initForwardTrain(4, 4, 4);

    n->copyInputOutputTrain(4, input, output);
    n->copyInputOutputValidation(4, input_val, output_val);

    n->trainAllExamplesMaxBatchSGD(20000, 500, 0.00000001, 0.0000001, 6, 0.01);

    n->trainGetCostFunctionAndCalculateLossFunction(4, 0);
    n->showForwardMatrices();

    n->finalizeForwardBackward();

    delete n;
    */

    /*
    int* nums = new int[20];
    for (int i = 0; i < 20; i++) { nums[i] = i; }
    
    printf("\n");
    for (int i = 0; i < 20; i++) { printf(" %d,", nums[i]); }
    printf("\n");

    edu_shuffle(nums, 20);

    printf("\n");
    for (int i = 0; i < 20; i++) { printf(" %d,", nums[i]); }
    printf("\n");
    */

    //FORWARD UNA RED VARIOS EJEMPLOS A LA VEZ

    /*
    Layer* l1 = new Layer(2, ELU, dELU);
    Layer* l2 = new Layer(1, Linear, dLinear);

    Network* n = new Network(2, 1, 2, new Layer * [2] {
        l1,
        l2
    }, MSE, dMSE);

    l1->copyWeightBias(new float[4] {-0.6057236802202783, -1.3743868186905905, 0.6057236802202786, 1.3743868186905905}, new float[2] {0.6057236802202781, -2.2751145817937323e-17});
    l2->copyWeightBias(new float[2] {-1.6689619673839928, 1.463146879527966}, new float[1] {1.0109297850315095});

    float* input = new float[4 * 2] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* output = new float[4 * 1] { 0, 1, 1, 0 };

    n->initForward(4);

    n->showWeightsBiasesLayers();

    float* res = new float[4];
    n->forward(4, input, res);

    n->showForwardMatrices();

    n->finalizeForwardBackward();

    delete n;
    */

    //ENTRENAMIENTO UNA SOLA RED

    /*

    Network* n = new Network(2, 1, 3, new Layer * [3] {
        new Layer(10, ELU, dELU),
        new Layer(10, ELU, dELU),
        new Layer(1, Linear, dLinear)
    }, MSE, dMSE);

    n->initWeightBiasValues();

    //n->showWeightsBiasesLayers();

    float* input = new float[4*2] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* output = new float[4*1] { 0, 1, 1, 0 };

    n->initForwardTrain(4, 0, 4);

    //n->showAuxiliarExpandReduceMatrices();

    n->copyInputOutputTrain(4, input, output);
    float err = 0;

    for (int i = 0; i < 10000; i++) {
        //n->showWeightsBiasesLayers();
        //printf("\n\nError MSE iteracion %d: %.20f\n", i + 1, n->backwardPhase(4, 4, new int[1] {0})[0]);
        //n->trainGetCostFunctionAndCalculateLossFunction(4, 4, new int[1] {0});
        //n->showForwardMatrices();
        //n->showErrorWeightsBiasesLayers();
        err = n->backwardPhase(4, 4, new int[1] {0})[0];
        if(i == 0 || (i+1)%500 == 0){ printf("\nError %d: %.20f", i+1, err); }
        n->applyVGradSGD(0.01);
    }

    n->forwardTrain(4);
    n->showForwardMatrices();

    n->finalizeForwardBackward();

    delete n;
    */

    //ENTRENAMIENTO VARIAS REDES

    /*

    Network* n = new Network(2, 5, 3, new Layer * [3] {
        new Layer(10, ELU, dELU),
        new Layer(10, ELU, dELU),
        new Layer(1, Linear, dLinear),
    }, MSE, dMSE);

    n->initWeightBiasValues();

    float* input = new float[4*2] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* output = new float[4*1] { 1, 0, 0, 1 };

    n->initForwardTrain(4, 0, 4);

    n->copyInputOutputTrain(4, input, output);
    float* errs;
    int* indx = new int[5] {0, 0, 0, 0, 0};

    n->trainGetCostFunctionAndCalculateLossFunction(4, 4, indx);
    n->showForwardMatrices();

    int niter = 20000;
    int mostrar_cada = 500;
    for (int i = 0; i < niter; i++) {
        errs = n->backwardPhase(4, 4, indx);
        if (i == 0 || (i + 1) % mostrar_cada == 0) { printf("\nErrores %d: %.20f %.20f %.20f %.20f %.20f", i + 1, errs[0], errs[1], errs[2], errs[3], errs[4]); }
        n->applyVGradSGD(0.01);
        delete errs;
    }

    n->trainGetCostFunctionAndCalculateLossFunction(4, 4, indx);
    n->showForwardMatrices();

    n->finalizeForwardBackward();

    delete n;
    */

    //PRUEBA FUNCIONAMIENTO FORWARD

    /*
    Layer* l1 = new Layer(2, ELU, dELU);
    Layer* l2 = new Layer(1, Linear, dLinear);

    Network* n = new Network(2, 2, 2, new Layer* [2]{
        l1,
        l2
    }, MSE, dMSE);

    l1->copyWeightBias(new float[8] {1.1172228789729295, 0.8939801347687951, 1.1172228787243454, 0.8939801345916509, 1.1172228789729295, 0.8939801347687951, 1.1172228787243454, 0.8939801345916509}, new float[4] {-1.1172228848589787, 5.448566789132996e-10, -1.1172228848589787, 5.448566789132996e-10});
    l2->copyWeightBias(new float[4] {-1.9048641164238775, 1.2619510820705655, -1.9048641164238775, 1.2619510820705655}, new float[2] {-0.12816004082232135, -0.12816004082232135});

    n->showInfoAboutNetwork();
    n->showWeightsBiasesLayers();

    float* res = new float[4];

    n->initForward(1);

    n->showAuxiliarExpandReduceMatrices();

    n->forward(1, new float[2 * 1] { 1, 1 }, res);

    n->showForwardMatrices();

    imprimirMatrizPorPantalla("Resultado forward host: ", res, 1, 1);

    n->finalizeForwardBackward();

    delete n;
    */

    //PRUEBA VELOCIDAD FORWARD

    /*
    Network* n = new Network(256, 2, 3, new Layer * [3] {
        new Layer(256, ELU, dELU),
        new Layer(256, ELU, dELU),
        new Layer(256, Linear, dLinear)
    });

    float* inp = new float[256];
    float* res = new float[256];

    n->initForward(1);

    std::chrono::time_point<std::chrono::system_clock> startCPU, endCPU;

    for (int i = 0; i < 100; i++) {
        startCPU = std::chrono::system_clock::now();

        n->forward(1, inp, res);

        endCPU = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = endCPU - startCPU;
        std::time_t end_time = std::chrono::system_clock::to_time_t(endCPU);

        std::cout << "elapsed time: " << elapsed_seconds.count() << "s; " << elapsed_seconds.count() * 1000 << "ms\n";
    }

    n->finalizeForwardBackward();

    delete n;
    */

    return 0;
}