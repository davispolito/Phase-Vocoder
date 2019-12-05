#pragma once
#include "kernel.h"
class PhaseVocoder{
private:
    const int sampling_rate = 44100;
    int nGroups; 	//nGroups - Number of groups of N samples
    const float beta = 6.8f;

public:
    float* imp;
	float* imp1;
	float* curr_input;
	float* prev_input;
	float* prev_output;
	float2* prev_mag_phase;
	float2* curr_mag_phase;
	cufftHandle plan;
	cufftHandle ifft;
    int hopSize;
    int nSamps;				//nSamps  - Length of original signal
    int R = 1; 				//R - Compression Ratio
    int N;	 				//N - num Phase Vocoder Channels 
		float timeScale;
		int outHopSize;

	void checkCUDAErrori(const char *msg, int line) {
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			if (line >= -1) {
				fprintf(stderr, "Line %d: ", line);
			}
			fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}

    PhaseVocoder(int samples): nSamps(samples), hopSize(samples/2), timeScale(1){
    	cudaMallocManaged((void**)&imp, sizeof(float)*samples, cudaMemAttachHost);
  		checkCUDAErrori("Malloc imp error",__LINE__);
    	cudaMallocManaged((void**)&imp1, sizeof(float)*samples, cudaMemAttachHost);
		  checkCUDAErrori("Malloc imp1 error",__LINE__);
    	cudaMallocManaged((void**)&prev_mag_phase, sizeof(float)* 2 * samples, cudaMemAttachHost);
		  checkCUDAErrori("Malloc prev_mag_phase error",__LINE__);
    	cudaMallocManaged((void**)&prev_input, sizeof(float) * samples, cudaMemAttachHost);
  		checkCUDAErrori("Malloc prev_input",__LINE__);
    	cudaMallocManaged((void**)&prev_output, sizeof(float) * samples, cudaMemAttachHost);
  		checkCUDAErrori("Malloc prev_output",__LINE__);
    	cudaMallocManaged((void**)&curr_mag_phase, sizeof(float) * samples, cudaMemAttachHost);
	  	checkCUDAErrori("Malloc curr_mag_phase",__LINE__);
    	//cudaMallocManaged((void**)&curr_input, sizeof(float) * samples, cudaMemAttachHost);
    	cudaMallocManaged((void**)&curr_input, sizeof(float) * samples, cudaMemAttachHost);
		  checkCUDAErrori("Malloc curr_input",__LINE__);
	   	//hanning window
		  //hanning window
	    for(int i = 0; i < samples; i++){
			  imp[i] = 0.5f * (1.f - cosf(2.f*M_PI*i / samples));
	    }
	    for(int i = 0; i < 2 * samples; i++){
	  		imp1[i] = 0.5f * (1.f - cosf(2.f*M_PI*i / samples));
	    }
	   	cufftPlan1d(&(this->plan),  2 * samples, CUFFT_C2C, 1);
		  checkCUDAErrori("Cufft Plan FFT Error",__LINE__);
		  cufftPlan1d(&(this->ifft), 2 * samples, CUFFT_C2R, 1);
		  checkCUDAErrori("Cufft Plan IFFT Error", __LINE__);
			outHopSize = hopSize * timeScale;
    }

    ~PhaseVocoder(){
	  cudaFree(imp);
    }
    void analysis();
    void analysis(float* input);
    void analysis(float* input, float2* output);
    void analysis(float* input, float2* output, int offset);
    void analysis(float* input, float2* output, float2* magFreq);
    void analysis(float* input, float2* output, float* intermediary);
    void resynthesis(float* backFrame, float2* frontFrame, float* output);
  	void resynthesis(float* backFrame, float2* frontFrame, float* output, void(*processing)());
};