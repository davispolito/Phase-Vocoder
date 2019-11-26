#include "kernel.h"
#include <math.h>
#include <cmath>
#include <stdio.h>
#define SAMPLING_FREQ 44100
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void cudaPVAnalysis(float* input, float2* output, float* imp, 
                          int impLen, int padZeros, int R, int N, int numSamps){
  int inputSample = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y  + threadIdx.y;
  
  inputSample *= R; 
  if (inputSample >= numSamps || i >= impLen){
    return; 
  }

  //Filter
}

__global__ void cudaFilter(float* input, float* imp,float* output, int impLen, int currSamp){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= impLen){
       return;
    }
    output[idx] = input[idx + currSamp] * imp[idx]; 
}

__global__ void cudaTimeAlias(float* preAlias, float* alias, int N, int nGroups){
  int idxS = blockIdx.x *blockDim.x + threadIdx.x;
  if(idxS >= N){
    return;
  }
  for(int i = 0; i < 2*nGroups; i++){
     alias[idxS] += preAlias[i*N + idxS];
  }
}
__global__ void cudaRotate(float2* shift, float* alias, int inc, int N){
  int idx = blockIdx.x *blockDim.x + threadIdx.x;
  if(idx >= N){
    return;
  }
  if(idx < inc){
    shift[idx].x = alias[N + idx - inc];
  } else {
    shift[idx].x = alias[idx - inc];
  }
}

__global__ void cudaFillOutput(float2* shift, 
  float2* output, int N){
  int idx = blockIdx.x *blockDim.x + threadIdx.x;
  if(idx > N / 2){
    return;
  }
  output[idx] = {shift[idx].x / (N/2), shift[idx].y / (N/2)};
}

__global__ void cudaPVAnalysis_rec(float2* output, float2* shift, 
  float* input, float* preAlias, float* alias, float* imp, 
  cufftHandle * plan,
  int impLen, int R, int N, int numSamps, int nGroups){
      int inputSample = blockIdx.x * blockDim.x + threadIdx.x;
      int oSample = inputSample;
      inputSample *= R; 
      if (inputSample >= numSamps){
          return;
      }
      
      //filter
      cudaFilter<<<1,impLen>>>(input, imp, &preAlias[oSample * impLen], impLen, inputSample);
      //time-aliasing
      cudaTimeAlias<<<1, N>>>(&preAlias[oSample * impLen], &alias[oSample * N], N, nGroups);
      //rotation
      int inc = inputSample % N;
      cudaRotate<<<1, N>>>(&shift[oSample * N], &alias[oSample * N], inc, N);
  }

__global__ void cufftShiftPadZeros(float2* output, float* input, int N, int numzeros){
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if(idx >= N / 2){
      return;
    }
    output[idx].x = input[idx + N/2];
    output[idx + N/2 + numzeros].x = input[idx];
}
__global__ void cufftShift(float2* output, float* input, int N){
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if(idx >= N / 2){
      return;
    }
    output[idx].x = input[idx + N/2];
    output[idx + N/2].x = input[idx];
}

__global__ void cudaWindow(float* input, float* win, int nSamps){ 
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if (idx >= nSamps){
        return;
    }
    input[idx] = input[idx] * win[idx];
  }
  

void ppArray(int n, float2 *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("{%3f, ", a[i].x);
        printf("%3f},\n", a[i].y);
    }
    printf("]\n");
}
void ppArray(int n, float *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3f \n ", a[i]);
    }
    printf("]\n");
}


  __global__ void cudaMagFreq(float2* output, float2* input, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N){
      return;
    }
    output[idx].x = sqrtf(input[idx].x * input[idx].x + input[idx].y * input[idx].y);
    output[idx].y = atanf(input[idx].y / input[idx].x);
  }
  namespace CudaPhase{
     void PVAnalysis(float2* output, float2* shift, float* input, float* preAlias, float* alias, float* imp, cufftHandle * plan, int impLen, int R, int N, int numSamps, int nGroups){
			cudaPVAnalysis_rec<<<1, N/R>>>(output, shift, input, preAlias, alias, imp, plan, impLen, R, N, numSamps, nGroups);
    }

      void pv_analysis(float2* output, float2* magFreq, float* input, float* win, int N, cufftHandle* plan){
          cudaWindow<<<1, N>>>(input, win, N);
          //cufftShift<<<1, N/2>>>(output, input, N);
          cufftShiftPadZeros<<<1, N/2>>>(output, input, N, N);
          cufftExecC2C(*plan, (cufftComplex *)output, (cufftComplex *)output, CUFFT_FORWARD);
          checkCUDAError("Cufft Error", __LINE__);
          cudaMagFreq<<<1, N>>>(magFreq, output, 2*N);
          
      }
  }
