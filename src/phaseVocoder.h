#pragma once
#include "kernel.h"
#define M_PI 3.141592653589793284
class PhaseVocoder{
private:
    const int sampling_rate = 44100;
    int nGroups; 	//nGroups - Number of groups of N samples
    const float beta = 6.8;

public:
    float* imp;
    int hopSize;
    int nSamps;		//nSamps  - Length of original signal
    int R = 1; 		//R - Compression Ratio
    int N;	 	//N - num Phase Vocoder Channels 
	cufftHandle plan;
	cufftHandle ifft;

    PhaseVocoder(int samples): nSamps(samples), hopSize(samples/2){
    	cudaMallocManaged((void**)&imp, sizeof(float)*samples, cudaMemAttachHost);	
		//hanning window
	    for(int i = 0; i < samples; i++){
			imp[i] = 0.5 * (1.f - cos(2.f*M_PI*i / samples));
	    }
		cufftPlan1d(&(this->plan),  2 * samples, CUFFT_C2C, 1);
		cufftPlan1d(&(this->ifft),  2 * samples, CUFFT_C2R, 1);


    }

    ~PhaseVocoder(){
	  cudaFree(imp);
    }
    void analysis(float* input, float2* output);
    void analysis(float* input, float2* output, float2* magFreq);
    void resynthesis(float* backFrame, float2* frontFrame, float* output);
	void resynthesis(float* backFrame, float2* frontFrame, float* output, void(*processing)());
};
