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
			output = shift;
	 }

void pArray(int n, float2 *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("{%3f, ", a[i].x);
        printf("%3f},", a[i].y);
    }
    printf("]\n");
}
void pArray(int n, float *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3f, ", a[i]);
    }
    printf("]\n");
}
void pArrayin0(int n, float2 *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        if(a[i].x != 0.f && a[i].y != 0.f){
        printf("{%3f, ", a[i].x);
        printf("%3f},", a[i].y);
        }
    }
    printf("]\n");
}
   void PhaseVocoder::analysis(float* input, float2* output, float2* magFreq){
       printf("input");
	   pArray(this->nSamps, input, true);
      cudaStreamAttachMemAsync(NULL, input, 0, cudaMemAttachGlobal);
      cudaStreamAttachMemAsync(NULL, magFreq, 0, cudaMemAttachGlobal);
      cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachGlobal);
      cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
	  CudaPhase::pv_analysis(output, magFreq, input, this->imp, this->nSamps, &this->plan);
      cudaStreamAttachMemAsync(NULL, magFreq, 0, cudaMemAttachHost);
      cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachHost);
      cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachHost);
	  cudaStreamSynchronize(NULL);
      printf("out\n");
      pArray(this->nSamps * 2, output, true);
      printf("mag\n");
      pArray(this->nSamps * 2, magFreq, true);
   }