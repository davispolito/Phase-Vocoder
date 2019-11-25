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
 void PVAnalysis(float2* output, float2* shift, float* input, float* preAlias, float* alias, float* imp, cufftHandle * plan, int impLen, int R, int N, int numSamps, int nGroups);
};