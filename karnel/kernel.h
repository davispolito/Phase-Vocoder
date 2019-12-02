#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include "cufft.h"
namespace CudaPhase{
 void pv_analysis(float2* output, float2* magFreq, float* input, float* win, int N, cufftHandle* plan);
 void resynthesis(float* output, float* backFrame, float2* frontFrame, float* win, int N, cufftHandle* plan, int hopSize);
};