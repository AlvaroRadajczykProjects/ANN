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

    /*
    Layer* l1 = new Layer(2, ELU, dELU);
    Layer* l2 = new Layer(1, Linear, dLinear);

    Network* n = new Network(2, 2, 2, new Layer* [2]{
        l1,
        l2
    });

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

    n->finalizeForward();

    delete n;
    */

    Layer* l1 = new Layer(2, ELU, dELU);
    Layer* l2 = new Layer(1, Linear, dLinear);

    Network* n = new Network(2, 2, 2, new Layer * [2] {
        l1,
        l2
    }, MSE, dMSE);

    l1->copyWeightBias(new float[8] {1.1172228789729295, 0.8939801347687951, 1.1172228787243454, 0.8939801345916509, 1.1172228789729295, 0.8939801347687951, 1.1172228787243454, 0.8939801345916509}, new float[4] {-1.1172228848589787, 0.0, -1.1172228848589787, 0.0});
    l2->copyWeightBias(new float[4] {-1.9048641164238775, 1.2619510820705655, -1.9048641164238775, 1.2619510820705655}, new float[2] {-0.12816004082232135, -0.12816004082232135});

    //n->showWeightsBiasesLayers();

    float* input = new float[4*2*2] { 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 1, 1, 0, 1, 1 };
    float* output = new float[4*1*2] { 9, 9, 9, 9, 0, 1, 1, 0 };

    n->initForwardTrain(8, 2);

    //n->showAuxiliarExpandReduceMatrices();

    n->copyInputOutputTrain(8, input, output);

    //n->forwardTrain(2);
    float err = n->trainGetCostFunctionAndCalculateLossFunction(8, 2, new int[2]{2, 3});

    printf("\n\nError MSE en host: %.16f\n", err);

    //n->showForwardMatrices();

    n->finalizeForward();

    delete n;

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

    n->finalizeForward();

    delete n;
    */

    return 0;
}