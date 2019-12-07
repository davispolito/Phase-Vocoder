#include "kernel.h"
#include "hpfft.h"
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
__global__ void cufftShiftPadZeros(float* output, float* input, int N, int numzeros){
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if(idx >= N / 2){
      return;
    }
    output[idx] = input[idx + N/2];
    output[idx + N/2 + numzeros]= input[idx];
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
  
__global__ void cudaWindow(float* input, float* output, float* win, int nSamps){ 
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if (idx >= nSamps){
        return;
    }
    output[idx] = input[idx] * win[idx];
  }
__global__ void cudaWindow(float* input, float* win, int nSamps){ 
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if (idx >= nSamps){
        return;
    }
    input[idx] = input[idx] * win[idx];
  }


  
__global__ void cudaWindow_HanRT(float* input, float* output, int nSamps){ 
    int idx = blockIdx.x *  blockDim.x + threadIdx.x; 
    if (idx >= nSamps){
        return;
    }
    output[idx] = input[idx] * 0.5f * (1.f - cosf(2.f*M_PI*idx / nSamps));
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
	if (idx >= N || idx < hopSize) {
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
    __global__ void cudaDivVec(float* input, float N, int scale) {

	  int idx = blockIdx.x * blockDim.x + threadIdx.x;
	  if (idx > N) {
		  return;
	  }
	  input[idx] /= scale;

  }
    __global__ void cudaDivVec(float2* input, float* output, float N) {

	  int idx = blockIdx.x * blockDim.x + threadIdx.x;
	  if (idx > (int)N) {
		  return;
	  }
	  output[idx] = input[idx].x / N;

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
  //#define DEBUGCUFFT
  //#define DEBUGMAG
  //#define DEBUGpad
  //#define DEBUGwindow
  namespace CudaPhase{
    ///usedbytwod
    
    
	  void pv_analysis(float2* output, float2* fft, float* input, float* intermediary, float* win, int N) {
      cudaWindow << <1, N >> > (input, intermediary, win, N);
		  checkCUDAError_("Window analysis", __LINE__);
         #ifdef DEBUGwindow
          float *debug_arr;
          cudaMallocManaged((void**)&debug_arr, sizeof(float) * N, cudaMemAttachHost);
      checkCUDAError_("Error debugging input after WINDOW (malloc)", __LINE__);
          cudaMemcpy(debug_arr,input, sizeof(float) * N,cudaMemcpyDeviceToHost);
      checkCUDAError_("Error debugging input after WINDOW (memcpy)", __LINE__);
          printf("in\n");
          printArraywNewLines(N, debug_arr);
          cudaMemcpy(debug_arr,intermediary, sizeof(float) * N,cudaMemcpyDeviceToHost);
      checkCUDAError_("Error debugging intermediary after WINDOW (mempy)", __LINE__);
          printf("intermediary\n");
          printArraywNewLines(N, debug_arr);
          cudaFree(debug_arr);
          #endif
		  cufftShiftPadZeros<<<1, N/2>>>(output, intermediary, N, N);
      checkCUDAError_("pad zero analysis", __LINE__);
         #ifdef DEBUGpad
          float2 *debug_arr1;
          cudaMallocManaged((void**)&debug_arr1, sizeof(float2) *2 * N, cudaMemAttachHost);
      checkCUDAError_("Error debugging output after cufftshift (malloc)", __LINE__);
          cudaMemcpy(debug_arr1,output, sizeof(float2) *2 * N,cudaMemcpyDeviceToHost);
      checkCUDAError_("Error debugging output after cufftshift (memcpy)", __LINE__);
          printf("out\n");
          printArraywNewLines(2*N, debug_arr1);
          cudaFree(debug_arr1);
          #endif
      FFT::HPFFT::computeGPUFFT(2*N, 2, output, fft);
		  checkCUDAError_("Cufft Error analysis", __LINE__);
         #ifdef DEBUGFFT
         float2 *debug_arr2;
          cudaMallocManaged((void**)&debug_arr2, sizeof(float2) * N, cudaMemAttachHost);
          cudaMemcpy(debug_arr2,output, sizeof(float2) * N,cudaMemcpyDeviceToHost);
          printf("postcufft\n");
          printArraywNewLines(N, debug_arr2);
          cudaFree(debug_arr2);
          #endif
		  cudaMagFreq << <1,2* N >> > (output,  2*N);
		  checkCUDAError_("magfreq Error analysis kernel.cu", __LINE__);
	  }
	  void pv_analysis_RT(float2* output, float2* fft, float* input, float* intermediary, float* win, int N, cudaStream_t* stream) {
      cudaWindow_HanRT<< <1, N , 0 , *stream>> > (input, intermediary, N);
		  checkCUDAError_("Window analysis", __LINE__);
         #ifdef DEBUGwindow
          float *debug_arr;
          cudaMallocManaged((void**)&debug_arr, sizeof(float) * N, cudaMemAttachHost);
      checkCUDAError_("Error debugging input after WINDOW (malloc)", __LINE__);
          cudaMemcpy(debug_arr,input, sizeof(float) * N,cudaMemcpyDeviceToHost);
      checkCUDAError_("Error debugging input after WINDOW (memcpy)", __LINE__);
          printf("in\n");
          printArraywNewLines(N, debug_arr);
          cudaMemcpy(debug_arr,intermediary, sizeof(float) * N,cudaMemcpyDeviceToHost);
      checkCUDAError_("Error debugging intermediary after WINDOW (mempy)", __LINE__);
          printf("intermediary\n");
          printArraywNewLines(N, debug_arr);
          cudaFree(debug_arr);
          #endif
		  cufftShiftPadZeros<<<1, N/2, 0, *stream>>>(output, intermediary, N, N);
      checkCUDAError_("pad zero analysis", __LINE__);
         #ifdef DEBUGpad
          float2 *debug_arr1;
          cudaMallocManaged((void**)&debug_arr1, sizeof(float2) *2 * N, cudaMemAttachHost);
      checkCUDAError_("Error debugging output after cufftshift (malloc)", __LINE__);
          cudaMemcpy(debug_arr1,output, sizeof(float2) *2 * N,cudaMemcpyDeviceToHost);
      checkCUDAError_("Error debugging output after cufftshift (memcpy)", __LINE__);
          printf("out\n");
          printArraywNewLines(2*N, debug_arr1);
          cudaFree(debug_arr1);
          #endif
      FFT::HPFFT::computeGPUFFT_RT(2*N, 2, output, fft, stream);
		  checkCUDAError_("Cufft Error analysis", __LINE__);
         #ifdef DEBUGFFT
         float2 *debug_arr2;
          cudaMallocManaged((void**)&debug_arr2, sizeof(float2) * N, cudaMemAttachHost);
          cudaMemcpy(debug_arr2,output, sizeof(float2) * N,cudaMemcpyDeviceToHost);
          printf("postcufft\n");
          printArraywNewLines(N, debug_arr2);
          cudaFree(debug_arr2);
          #endif
		  cudaMagFreq << <1,2* N, 0,*stream >> > (output,  2*N);
		  checkCUDAError_("magfreq Error analysis kernel.cu", __LINE__);
	  }
    //#define DEBUGIFFT
		void resynthesis(float* output, float* backFrame, float2* frontFrame, float2* intermediary, float* win, int N,  int hopSize) {
		 	cudaTimeScale << <1, 2* N >> > (frontFrame,2* N, 1);

      FFT::HPFFT::computeGPUIFFT(2*N, 2, frontFrame, intermediary);
			checkCUDAError_("ifft error");
         #ifdef DEBUGIFFT
         float2 *debug_arr2;
          cudaMallocManaged((void**)&debug_arr2, sizeof(float2) * 2 * N, cudaMemAttachHost);
          cudaMemcpy(debug_arr2,intermediary, sizeof(float2) * 2 * N,cudaMemcpyDeviceToHost);
          printf("postcufft\n");
          printArraywNewLines(N, debug_arr2);
          cudaFree(debug_arr2);
          #endif

			cudaDivVec << <1, N >> > ( intermediary,output, N);
			checkCUDAError_("divvec error");

			cufftShift<<<1,N/2>>>(output, N);
			checkCUDAError_("shift error");

			cudaWindow<<<1, N>>>(output, win,N);
			checkCUDAError_("window error");

			cudaOverlapAdd<<<1,N>>>(backFrame, output, N, hopSize);
			checkCUDAError_("add error");
	  
      }
      void test_overlap_add(float* input, float* output, float* intermediary, float* backFrame, float* win, int N, int hopSize){
          cudaWindow<< <1,N >> > (input, intermediary, win, N);
			checkCUDAError_("window error", __LINE__);
		      cufftShift<<<1, N/2>>>(output, intermediary, N, N);
			checkCUDAError_("shift error", __LINE__);
			    cufftShift<<<1,N/2>>>(output, N);
			checkCUDAError_("shift error", __LINE__);
			    cudaWindow<<<1, N>>>(output, win, N);
			    cudaOverlapAdd<<<1,N>>>(backFrame, output, N, hopSize);
      }
	  void pv_analysis_CUFFT(float2* output, float2* fft, float* input, float* intermediary, float* win, int N) {
      cudaWindow<< <1,N >> > (input, intermediary, win, N);
         #ifdef DEBUGwindow
          float *debug_arr;
          cudaMallocManaged((void**)&debug_arr, sizeof(float) * N, cudaMemAttachHost);
          cudaMemcpy(debug_arr,input, sizeof(float) * N,cudaMemcpyDeviceToHost);
          printf("in\n");
          printArraywNewLines(N, debug_arr);
          cudaMemcpy(debug_arr,intermediary, sizeof(float) * N,cudaMemcpyDeviceToHost);
          printf("intermediary\n");
          printArraywNewLines(N, debug_arr);
          cudaFree(debug_arr);
          #endif
		  checkCUDAError_("Window analysis", __LINE__);
		  cufftShiftPadZeros<<<1, N/2>>>(output, intermediary, N, N);
         #ifdef DEBUGpad
          float2 *debug_arr1;
          cudaMallocManaged((void**)&debug_arr1, sizeof(float2) * N, cudaMemAttachHost);
          cudaMemcpy(debug_arr1,output, sizeof(float2) * N,cudaMemcpyDeviceToHost);
          printf("out\n");
          printArraywNewLines(N, debug_arr1);
          cudaFree(debug_arr1);
          #endif
      checkCUDAError_("pad zero analysis", __LINE__);
      cufftHandle plan;
		  cufftPlan1d(&plan, 2 * N, CUFFT_C2C, 1);
		  cufftExecC2C(plan, (cufftComplex *)output, (cufftComplex *)output, CUFFT_FORWARD);
		  checkCUDAError_("Cufft Error analysis", __LINE__);
         #ifdef DEBUGCUFFT
         float2 *debug_arr2;
          cudaMallocManaged((void**)&debug_arr2, sizeof(float2) *2* N, cudaMemAttachHost);
          cudaMemcpy(debug_arr2,output, sizeof(float2) *2* N,cudaMemcpyDeviceToHost);
          printf("postcufft\n");
          printArraywNewLines(2*N, debug_arr2);
          cudaFree(debug_arr2);
          #endif
      cufftDestroy(plan);
      cudaMagFreq << <1, 2*N >> > (output,  2*N);
       #ifdef DEBUGMAG
         float2 *debug_arr3;
          cudaMallocManaged((void**)&debug_arr3, sizeof(float2) *2 * N, cudaMemAttachHost);
          cudaMemcpy(debug_arr3,output, sizeof(float2) *2 * N,cudaMemcpyDeviceToHost);
          printf("postMagnitude\n");
          printArraywNewLines(2*N, debug_arr3);
          cudaFree(debug_arr3);
          #endif
		  checkCUDAError_("magfreq Error analysis", __LINE__);
    }
      //#define DEBUGTS
      //#define DEBUGIFFT
      //#define DEBUGSHIFTRE
		void resynthesis_CUFFT(float* output, float* backFrame, float2* frontFrame, float* win,int N, int hopSize) {
		 	cudaTimeScale << <1,2* N >> > (frontFrame,2* N, 1);
         #ifdef DEBUGTS
         float2 *debug_arr2;
          cudaMallocManaged((void**)&debug_arr2, sizeof(float2) * 2 * N, cudaMemAttachHost);
          cudaMemcpy(debug_arr2,frontFrame, sizeof(float2) * 2 * N,cudaMemcpyDeviceToHost);
          printf("postTS\n");
          printArraywNewLines(N, debug_arr2);
          cudaFree(debug_arr2);
          #endif
      cufftHandle plan;
		  cufftPlan1d(&plan,  N, CUFFT_C2R, 1);
		  checkCUDAError_("Cufft Plan IFFT Error", __LINE__);
			cufftExecC2R(plan, (cufftComplex*)frontFrame, (cufftReal *)output);
			checkCUDAError_("ifft error");
         #ifdef DEBUGIFFT
         float *debug_arr;
          cudaMallocManaged((void**)&debug_arr, sizeof(float) *  N, cudaMemAttachHost);
          checkCUDAError_("Error debugging output after ifft (malloc)", __LINE__);
          cudaMemcpy(debug_arr,output, sizeof(float) * N,cudaMemcpyHostToHost);
          checkCUDAError_("Error debugging output after ifft (memcpy)", __LINE__);
          printf("CU IFFT\n");
          printArraywNewLines(N, debug_arr);
          cudaFree(debug_arr);
          #endif
      cufftDestroy(plan);
			checkCUDAError_("cufftDestory error");
			cudaDivVec << <1, N >> > (output, N, N);
      checkCUDAError_("divvec error");
      #ifdef DEBUGSCALE
         float *debug_arr1;
          cudaMallocManaged((void**)&debug_arr1, sizeof(float) * N, cudaMemAttachHost);
          checkCUDAError_("Error debugging output after ifft (malloc)", __LINE__);
          cudaMemcpy(debug_arr1, output, sizeof(float) * N,cudaMemcpyHostToHost);
          checkCUDAError_("Error debugging output after ifft (memcpy)", __LINE__);
          printf("SCALE RE\n");
          printArraywNewLines(N, debug_arr1);
          cudaFree(debug_arr1);
      #endif

			cufftShift<<<1,N/2>>>(output, N);
			checkCUDAError_("shift error");
      #ifdef DEBUGSHIFTRE
         float *debug_arr3;
          cudaMallocManaged((void**)&debug_arr3, sizeof(float) * N, cudaMemAttachHost);
          checkCUDAError_("Error debugging output after ifft (malloc)", __LINE__);
          cudaMemcpy(debug_arr3, output, sizeof(float) * N,cudaMemcpyHostToHost);
          checkCUDAError_("Error debugging output after ifft (memcpy)", __LINE__);
          printf("SHIFT RE\n");
          printArraywNewLines(N, debug_arr3);
          cudaFree(debug_arr3);
      #endif

			cudaWindow<<<1, N>>>(output, win, N);
			checkCUDAError_("window error");
      #ifdef DEBUGSHIFTRE
         float *debug_arr4;
          cudaMallocManaged((void**)&debug_arr4, sizeof(float) * N, cudaMemAttachHost);
          checkCUDAError_("Error debugging output after ifft (malloc)", __LINE__);
          cudaMemcpy(debug_arr4, output, sizeof(float) * N,cudaMemcpyHostToHost);
          checkCUDAError_("Error debugging output after ifft (memcpy)", __LINE__);
          printf("WINDOW resynth\n");
          printArraywNewLines(N, debug_arr4);
          cudaFree(debug_arr4);
      #endif

			cudaOverlapAdd<<<1,N>>>(backFrame, output, N, hopSize);
			checkCUDAError_("add error");
      #ifdef DEBUGOADD
         float *debug_arr5;
          cudaMallocManaged((void**)&debug_arr5, sizeof(float) * N, cudaMemAttachHost);
          checkCUDAError_("Error debugging output after ifft (malloc)", __LINE__);
          cudaMemcpy(debug_arr5, output, sizeof(float) * N,cudaMemcpyHostToHost);
          checkCUDAError_("Error debugging output after ifft (memcpy)", __LINE__);
          printf("WINDOW resynth\n");
          printArraywNewLines(N, debug_arr5);
          cudaFree(debug_arr5);
      #endif
	  
      }
  }