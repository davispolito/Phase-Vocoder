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
 void pv_analysis(float2* output,float2* fft, float* input, float * intermediary, float* win, int N);
 void pv_analysis_RT(float2* output,float2* fft, float* input, float * intermediary,int N, cudaStream_t* stream);
 void pv_analysis_CUFFT(float2* output,float2* fft, float* input, float * intermediary, float* win, int N);
 void resynthesis(float* output, float* backFrame, float2* frontFrame, float2* intermediary, float* win, int N, int hopSize);
 void resynthesis_CUFFT(float* output, float* backFrame, float2* frontFrame, float* win,int N, int hopSize);
};