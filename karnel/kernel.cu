#include "kernel.h"
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <cuda.h>
#include "kernel.h"
#define M_PI 3.1415926535897931
//divide
#define THREADS_PER_SAMPLE 16
#define SAMPLES_PER_THREAD 1
#define SAMPLING_FREQ 44100
//#define SIMPLE 0
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
float* dev_frequencies, *dev_buffer, *dev_tmp_buffer, *dev_gains, *dev_target, *dev_angle;
float slideTime;
int numSamples, numSinusoids;

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


void Additive::initSynth(int numSinusoid, int numSample, float* host_frequencies) {
	
		numSamples = numSample;
		numSinusoids = numSinusoid;
		cudaMalloc((void**)&dev_frequencies, numSinusoids * sizeof(float));
		cudaMalloc((void**)&dev_buffer, numSamples * sizeof(float));
		cudaMemcpy(dev_frequencies, host_frequencies, numSinusoids * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaDeviceSynchronize();
}

void Additive::initSynth_THX(int numSinusoid, int numSample, float* host_start_freq, float* host_end_freq, float* host_angle, float*  host_gains, float slide) {
	numSamples = numSample;
	numSinusoids = numSinusoid;
	slideTime = slide;
	cudaMalloc((void**)&dev_frequencies, numSinusoids * sizeof(float));
	checkCUDAErrorWithLine("dev_frequencies malloc failed");
	cudaMemcpy(dev_frequencies, host_start_freq, numSinusoids * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("dev_frequencies memcpy failed");
	cudaMalloc((void**)&dev_buffer, numSamples * sizeof(float));
	checkCUDAErrorWithLine("dev_buffer malloc failed");
	cudaMalloc((void**)&dev_tmp_buffer, numSamples *THREADS_PER_SAMPLE* sizeof(float));
	checkCUDAErrorWithLine("dev_tmp_buffer malloc failed");
	cudaMalloc((void**)&dev_gains, numSinusoids * sizeof(float));
	checkCUDAErrorWithLine("dev_gains malloc failed");
	cudaMemcpy(dev_gains, host_gains, numSinusoids * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("dev_gains memcpy failed");
	cudaMalloc((void**)&dev_angle, numSinusoids * sizeof(float));
	checkCUDAErrorWithLine("dev_angle malloc failed");
	//cudaMemcpy(dev_angle, host_angle, numSinusoids * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_target, numSinusoids * sizeof(float));
	checkCUDAErrorWithLine("dev_target malloc failed");
	cudaMemcpy(dev_target, host_end_freq, numSinusoids * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("target frequencies copy failed");
	cudaDeviceSynchronize();
}

void Additive::endSynth_THX() {
	cudaFree(dev_frequencies);
	cudaFree(dev_buffer);
	cudaFree(dev_tmp_buffer);
	cudaFree(dev_gains);
	cudaFree(dev_angle);
	cudaFree(dev_target);
}
void Additive::endSynth() {
	cudaFree(dev_buffer);
	cudaFree(dev_frequencies);
}
__global__ void sin_kernel_simple(float * buffer, float* frequencies, float angle, int numSamples, int numSinusoids) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numSamples) {
		angle = angle + 2.0f * M_PI * idx / 44100.f;
		float val = 0.0f;
		for (int i = 0; i < numSinusoids; i++){
			val +=  0.1 * __sinf((angle * frequencies[i]));
		}
		buffer[idx] = val;
	}
}

void Additive::compute_sinusoid_gpu_simple(float* buffer, float angle) {
	int threadsPerBlock = 256; 
	int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

	sin_kernel_simple << <blocksPerGrid, threadsPerBlock >> > (dev_buffer, dev_frequencies, angle, numSamples, numSinusoids);
//	sin_kernel_simple << <1, 256>> > (dev_buffer, dev_frequencies, angle, numSamples, numSinusoids);
	
	cudaMemcpy(buffer, dev_buffer, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
}

__device__ float ramp_kern(float currentTime, float slideTime, float f0, float f1){
	float integral;
	if (currentTime < slideTime) {
		float k = (f1-f0) / slideTime;
		integral = currentTime * (f0 + k * currentTime / 2.0f);
	} else {
		integral = f0 * slideTime + (f1 - f0) * slideTime / 2.0f;
		integral += (currentTime - slideTime) * f1;
	}
	return integral * 2.0f * M_PI;
}

#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )

__global__ void sin_kernel_fast(float * buffer, float * frequencies, float* targetFrequencies, float* angles, float* gains, int numThreadsPerBlock, int numSinusoids,
	float time, float slideTime, int numSamples) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numSamples * THREADS_PER_SAMPLE) {
		//determine how many sineWaves are to be computed in each thread based on how many threads it takes to compute a sample
		int maxSinePerBlock = (numSinusoids + THREADS_PER_SAMPLE - 1) / THREADS_PER_SAMPLE;
		int sinBlock = idx / numThreadsPerBlock;
		int sampleIdx = idx - sinBlock * numThreadsPerBlock; // modulo function but GPUs are trash at modulo so don't use it
		float val[SAMPLES_PER_THREAD];
		for (int j = 0; j < SAMPLES_PER_THREAD; j++) {
			val[j] = 0.0f;
		}
	    float gain, freq0, freq1, angle, angleStart;
	    int firstSine = sinBlock * maxSinePerBlock;
		int lastSine = imin(numSinusoids, firstSine + maxSinePerBlock);
		//compute samples for maxSinePerBlock
		for (int i = firstSine; i < lastSine; i++) {
			angleStart = 0; 
			freq0 = frequencies[i];
			freq1 = targetFrequencies[i];
			gain = gains[i];
			for (int j = 0; j < SAMPLES_PER_THREAD; j++) {
				angle = ramp_kern(time + (sampleIdx * SAMPLES_PER_THREAD + j) / SAMPLING_FREQ, slideTime, freq0, freq1);
				val[j] += __sinf(angleStart + angle) * gain / numSinusoids;
			}
		}
		for (int i = 0; i < SAMPLES_PER_THREAD; i++) {
			buffer[idx * SAMPLES_PER_THREAD + i] = val[i];
		}

	}

}


__global__ void sum_blocks(float* tmp_buffer, float* buffer, int numSamples) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numSamples) {
		float sum = 0;
		for (int i = 0; i < THREADS_PER_SAMPLE; i++) {
			sum += tmp_buffer[idx + i * numSamples];
		}
		buffer[idx] = sum;
	}

}

void Additive::compute_sinusoid_hybrid(float* buffer, float * time){
	int threadsPerBlock = 256; 
	int numThreadsPerBlock = numSamples / SAMPLES_PER_THREAD;
	int numThreads = THREADS_PER_SAMPLE * numThreadsPerBlock;
	int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

	
	sin_kernel_fast <<<blocksPerGrid, threadsPerBlock >>>(dev_tmp_buffer, dev_frequencies, dev_target, dev_angle, dev_gains, numThreadsPerBlock, numSinusoids, *time, slideTime, numSamples);
	checkCUDAErrorWithLine("sin_kernel_fast failed");
	blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
	sum_blocks <<<blocksPerGrid, threadsPerBlock >> >(dev_tmp_buffer, dev_buffer, numSamples);
	checkCUDAErrorWithLine("sum_blocks failed");
	cudaMemcpy(buffer, dev_buffer, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
}



