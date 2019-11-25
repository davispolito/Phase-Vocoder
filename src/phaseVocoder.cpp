#include "phaseVocoder.h"
   void PhaseVocoder::analysis(float* input, float2* output){
			int impLen = 2 * N * nGroups + this->nSamps - 1; 	// length of impulse response of lowpass filter
			int padZeros;
			int inc;	//numsamples to rotate Alias
			int iSampNo; 	//index for input points in array X at og sampling rate
			int oSampNo;    //index for output points art samplingrate Srate/R
			int i,k;
			float *X; //signal padded by 0s

			cudaMallocManaged((void**)&X, sizeof(float)*impLen);
			memset(X, 0, impLen);
			memcpy((void*)&X[N*nGroups], (void*)input, sizeof(float) * this->nSamps);
			//filter input signal
	 		float * alias;
			cudaMallocManaged((void**)&alias, sizeof(float)*N * N/R);	
	 		float * preAlias;
			cudaMallocManaged((void**)&preAlias, sizeof(float)*impLen * N /R);	
	 		float2 * shift;
			cudaMallocManaged((void**)&shift, sizeof(float2)*N * N/R);	
      cudaStreamAttachMemAsync(NULL, shift, 0, cudaMemAttachGlobal);
			CudaPhase::PVAnalysis(output, shift, X, preAlias, alias, this->imp, &this->plan, 
			impLen, this->R, this->N, this->nSamps, this->nGroups);
			for(int i = 0; i < this->N / this->R; i++){
      	cufftExecC2C(this->plan, (cufftComplex *)&shift[i * N], (cufftComplex *)&shift[i * N], CUFFT_FORWARD);
			}

	 }