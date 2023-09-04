#include <stdio.h>
#include <chrono>

#include "ActivationLossFunctions.cuh"
#include "Network.cuh"

#define N 20

using namespace std;

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

//first epoch number is 0
float lrate_func(int epoch) {
    return 0.001;
}

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

    int nrand = (rand() % 300) + 100;
    int nrand2 = (rand() % 300) + 100;

    int temp = max(nrand, nrand2);
    int temp2 = min(nrand, nrand2);

    nrand = temp;
    nrand2 = temp2;

    printf("Random number: %d\n", nrand);
    printf("Random number 2: %d\n", nrand2);

    Network* n = new Network(256, 1, 3, new Layer * [3] {
        new Layer(256, ELU, dELU),
        new Layer(256, ELU, dELU),
        new Layer(256, Linear, dLinear),
    }, MSE, dMSE);

    n->initWeightBiasValues();

    float* input = new float[256 * nrand];
    float* output = new float[256 * nrand];

    float* input2 = new float[256 * nrand2];
    float* output2 = new float[256 * nrand2];

    for (int i = 0; i < 256 * nrand; i++) { input[i] = 100; output[i] = 2; }
    for (int i = 0; i < 256 * nrand2; i++) { input2[i] = 100; output2[i] = 2; }

    n->initForwardTrain(nrand, nrand2, 32);

    n->copyInputOutputTrain(nrand, input, output);
    n->copyInputOutputValidation(nrand2, input2, output2);

    n->trainAllExamplesMaxBatch(lrate_func, NULL, 0, applyVGradSGD, 10000, 500, 0.1, 0.0001, 6);

    n->showForwardMatrices();

    n->finalizeForwardBackward();

    delete n;

    /*
    Network* n = new Network(256, 1, 3, new Layer * [3] {
        new Layer(256, ELU, dELU),
            new Layer(256, ELU, dELU),
            new Layer(256, Linear, dLinear),
        }, MSE, dMSE);

    n->loadNetworkFromFile("network.data");

    float* input = new float[256 * 144];
    float* output = new float[256 * 144];

    for (int i = 0; i < 256 * 144; i++) { input[i] = 10; output[i] = 0; }

    n->initForwardTrain(144, 144, 32);

    n->copyInputOutputTrain(144, input, output);
    n->copyInputOutputValidation(144, input, output);

    n->trainGetCostFunctionAndCalculateLossFunction(32, 0);
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

    l1->copyWeightBias(new float[8] {-0.6057236802202783, -1.3743868186905905, 0.6057236802202786, 1.3743868186905905, -0.6057236802202783, -1.3743868186905905, 0.6057236802202786, 1.3743868186905905}, new float[4] {0.6057236802202781, -2.2751145817937323e-17, 0.6057236802202781, -2.2751145817937323e-17});
    l2->copyWeightBias(new float[4] {-1.6689619673839928, 1.463146879527966, -1.6689619673839928, 1.463146879527966}, new float[2] {1.0109297850315095, 1.0109297850315095});

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

    imprimirMatrizPorPantalla("Resultado forward host: ", res, 4, 1);

    n->finalizeForwardBackward();

    delete n;
    */

    //PRUEBA VELOCIDAD FORWARD

    /*
    Network* n = new Network(256, 2, 3, new Layer * [3] {
        new Layer(256, ELU, dELU),
        new Layer(256, ELU, dELU),
        new Layer(256, Linear, dLinear)
    }, MSE, dMSE);

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
        //err = n->backwardPhase(4, 0, NULL)[0];
        err = n->backwardPhase(4, 0, NULL)[0];
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

    int niter = 20000;
    int mostrar_cada = 500;
    for (int i = 0; i < niter; i++) {
        errs = n->backwardPhase(4, 0, NULL);
        if (i == 0 || (i + 1) % mostrar_cada == 0) { printf("\nErrores %d: %.20f %.20f %.20f %.20f %.20f", i + 1, errs[0], errs[1], errs[2], errs[3], errs[4]); }
        n->applyVGradSGD(0.01);
        delete errs;
    }

    n->trainGetCostFunctionAndCalculateLossFunction(4, 0);
    n->showForwardMatrices();

    n->finalizeForwardBackward();

    delete n;
    */

    /*
    HostPinnedDeviceMatrix* p = new HostPinnedDeviceMatrix(3, 9, 4, cudaHostAllocWriteCombined);

    float* misdatos = new float[3 * 9];
    float* misdatos_cop = new float[3 * 9];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 9; j++) {
            misdatos[i * 9 + j] = (j+1) * (i+1);
        }
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    imprimirMatrizPorPantalla("a", misdatos, 3, 9);
    imprimirMatrizPorPantalla("b", misdatos_cop, 3, 9);

    p->showDeviceMatrix("c", stream);
    p->copyHostToDevice(misdatos, 4, stream);
    cudaStreamSynchronize(stream);
    p->showDeviceMatrix("d", stream);

    imprimirMatrizPorPantalla("e", misdatos, 3, 9);
    imprimirMatrizPorPantalla("f", misdatos_cop, 3, 9);

    p->showDeviceMatrix("g", stream);
    p->copyDeviceToHost(misdatos_cop, stream);
    cudaStreamSynchronize(stream);
    p->showDeviceMatrix("h", stream);

    imprimirMatrizPorPantalla("i", misdatos, 3, 9);
    imprimirMatrizPorPantalla("j", misdatos_cop, 3, 9);

    float** d_pointers = p->generateDeviceRowsPointers(0, 3, new int[3] {0, 1, 2});
    float** hd_pointers = new float* [3];
    cudaMemcpy(hd_pointers, d_pointers, 3*sizeof(float*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 3; i++) {
        float* hrow = new float[9];
        cudaMemcpy(hrow, hd_pointers[i], 9*sizeof(float), cudaMemcpyDeviceToHost);
        imprimirMatrizPorPantalla("vector xd: ", hrow, 1, 9);
        delete hrow;
    }

    float* d_copy = 0;
    cudaMalloc(&d_copy, 3*9*sizeof(float));
    p->copyToDevice(d_copy, 3, stream);

    float* h_copy = new float[3 * 9];
    cudaMemcpy(h_copy, d_copy, 3 * 9 * sizeof(float), cudaMemcpyDeviceToHost);

    imprimirMatrizPorPantalla("h_copy: ", h_copy, 3, 9);

    //cudaFree(d_copy);
    cudaMalloc(&d_copy, 3 * 9 * sizeof(float));

    h_copy = new float[3 * 9];
    cudaMemcpy(h_copy, d_copy, 3 * 9 * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("h_copy thrash: ", h_copy, 3, 9);
    delete h_copy;

    p->copyFromDevice(d_copy, 3, stream);

    p->showDeviceMatrix("final", stream);

    delete p;
    */

    return 0;
}