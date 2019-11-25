#pragma once
#include "KaiserWindow.h"
#include "kernel.h"
class PhaseVocoder{
private:
    const int sampling_rate = 44100;
    int nGroups; 	//nGroups - Number of groups of N samples
    int nSamps;		//nSamps  - Length of original signal
    const float beta = 6.8;
    float* imp;
	cufftHandle plan;

public:
    int R = 1; 		//R - Compression Ratio
    int N;	 	//N - num Phase Vocoder Channels 
    PhaseVocoder():N(512), nGroups(4.0), nSamps(256){
	    int winLen = N*nGroups + 1;
	    int impLen = 2*winLen -1;
	    std::vector<float> win(impLen);
    	cudaMallocManaged((void**)&imp, sizeof(float)*nGroups*N, cudaMemAttachHost);	
	    Loris::KaiserWindow::buildWindow(win, (float)6.8);
	    for(int i = 0; i < nGroups * N; i++){
	       imp[i] = N * win[i] * sin(M_PI * i / N) / (M_PI * i);
	    }
		cufftPlan1d(&(this->plan), this->N * sizeof(float2*), CUFFT_C2C, 1);
    }

    ~PhaseVocoder(){
	  cudaFree(imp);
    }
    void analysis(float* input, float2* output);
};
