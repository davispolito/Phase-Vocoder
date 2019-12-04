#include "kernel.h"
#include <math.h>
#include <cmath>
#include <stdio.h>
#include "../src/io.h"
#define SAMPLING_FREQ 44100

__global__ void cufftShiftPadZeros(float2* output, float* input, int N, int numzeros, int offset){
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if(idx >= N / 2){
      return;
    }
    output[idx].x = input[(idx + offset) + N/2];
    output[idx + N/2 + numzeros].x = input[idx + offset];
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

__global__ void cufftShift(float* output, float* input, int N){
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if(idx >= N / 2){
      return;
    }
    output[idx] = input[idx + N/2];
    output[idx + N/2] = input[idx];
}

__global__ void cufftShift(float* input, int N){
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if(idx >= N / 2){
      return;
    }
	float tmp = input[idx];
    input[idx] = input[idx + N/2];
    input[idx + N/2] = tmp;
}
__global__ void cudaWindow(float* input, float* win, int nSamps, int offset){ 
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if (idx >= nSamps){
        return;
    }
    input[idx + offset] = input[idx + offset] * win[idx];
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

  __global__ void cudaMagFreq(float2* input, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N){
      return;
    }
	float2 temp = input[idx];
    input[idx].x = sqrtf(temp.x * temp.x + temp.y * temp.y);
    input[idx].y = atanf(temp.y / temp.x);
  }

  __global__ void cudaOverlapAdd(float* backFrame, float* frontFrame, int N, int hopSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N && idx >= hopSize) {
		return;
	}

	frontFrame[idx - hopSize] += backFrame[idx];

  }

  __global__ void  cudaTimeScale(float2* input, int N, int timeScale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >  N) {
		return;
	}

	input[idx].x = input[idx].x * cosf(timeScale * input[idx].y);
	input[idx].y = input[idx].x * sinf(timeScale * input[idx].y);
   }
    __global__ void cudaDivVec(float* input, int N, int scale) {

	  int idx = blockIdx.x * blockDim.x + threadIdx.x;
	  if (idx > N) {
		  return;
	  }
	  input[idx] /= scale;

  }

	__global__ void padZeros(float* input, float2* output, int N, int zeros) {
	  int idx = blockIdx.x * blockDim.x + threadIdx.x;
	  if (idx > N + zeros) {
		  return;
	  }
	  if (idx > zeros / 2 && idx < zeros / 2 + N) {
		  output[idx].x = input[idx - (zeros / 2)];
	  }
	  else {
		  output[idx].x = 0;
	  }
		

	}
  namespace CudaPhase{

	  void pv_analysis(float2* output, float* input, float* win, int N, cufftHandle* plan, int offset) {
      cudaWindow << <1, N >> > (input, win, N, offset);
      cudaStreamSynchronize(NULL);
		  checkCUDAError_("Window analysis", __LINE__);
		  cufftShiftPadZeros<<<1, N/2>>>(output, input, N, N, offset);
		  checkCUDAError_("pad zero analysis", __LINE__);
		  cufftExecC2C(*plan, (cufftComplex *)output, (cufftComplex *)output, CUFFT_FORWARD);
		  checkCUDAError_("Cufft Error analysis", __LINE__);
		  cudaMagFreq << <1, N >> > (output,  N);
		  checkCUDAError_("magfreq Error analysis", __LINE__);
	  }
	  void pv_analysis(float2* output, float* input, float* win, int N, cufftHandle* plan) {
		  cudaWindow << <1, N >> > (input, win, N);
		  checkCUDAError_("Window analysis", __LINE__);
		  cufftShiftPadZeros<<<1, N/2>>>(output, input, N, N);
		  checkCUDAError_("pad zero analysis", __LINE__);
		  cufftExecC2C(*plan, (cufftComplex *)output, (cufftComplex *)output, CUFFT_FORWARD);
		  checkCUDAError_("Cufft Error analysis", __LINE__);
		  cudaMagFreq << <1, N >> > (output,  N);
		  checkCUDAError_("magfreq Error analysis", __LINE__);
	  }
	  void pv_analysis(float2* output, float2* magFreq, float* input, float* win, int N, cufftHandle* plan) {
		  cudaWindow << <1, N >> > (input, win, N);
	cudaStreamSynchronize(NULL);
		  checkCUDAError_("Window analysis", __LINE__);
		  cufftShiftPadZeros<<<1, N/2>>>(output, input, N, N);
	cudaStreamSynchronize(NULL);
		  checkCUDAError_("pad zero analysis", __LINE__);
		  cufftExecC2C(*plan, (cufftComplex *)output, (cufftComplex *)output, CUFFT_FORWARD);
	cudaStreamSynchronize(NULL);
		  checkCUDAError_("Cufft Error analysis", __LINE__);
		  cudaMagFreq << <1, N >> > (magFreq, output,  N);
	cudaStreamSynchronize(NULL);
		  checkCUDAError_("magfreq Error analysis", __LINE__);
	  }
		void resynthesis(float* output, float* backFrame, float2* frontFrame, float* win, int N, cufftHandle* plan, int hopSize) {
		 	cudaTimeScale << <1, N >> > (frontFrame, N, 1);

			cufftExecC2R(*plan, frontFrame, output);
			checkCUDAError_("ifft error");

			cudaDivVec << <1,2 * N >> > (output,2 * N, N);
			checkCUDAError_("divvec error");

			cufftShift<<<1,N>>>(output, 2*N);
			checkCUDAError_("shift error");

			cudaWindow<<<1,2* N>>>(output, win,2* N);
			checkCUDAError_("window error");

			cudaOverlapAdd<<<1,N>>>(backFrame, output, N, hopSize);
			checkCUDAError_("add error");
	  
      }
  }