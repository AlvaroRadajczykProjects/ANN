#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <cassert>

using func_t = float(*) (float);
using func2_t = float(*) (float, float);
using func3_t = float(*) (float, float, float);

__global__ void applyFunctionVectorial(float* arr, func_t func);