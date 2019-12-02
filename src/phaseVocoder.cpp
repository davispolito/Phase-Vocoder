#include "phaseVocoder.h"
#include <cstdlib>
#include "io.h"
using namespace std;

void PhaseVocoder::analysis(float* input, float2* output, float2* magFreq){
	//cudaStreamAttachMemAsync(NULL, input, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach input", __LINE__);
    cudaStreamAttachMemAsync(NULL, magFreq, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach mag", __LINE__);
    cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach out", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach win", __LINE__);
	cudaStreamSynchronize(NULL);
	CudaPhase::pv_analysis(output, magFreq, input, this->imp, this->nSamps, &this->plan);
    cudaStreamAttachMemAsync(NULL, magFreq, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}

void PhaseVocoder::resynthesis(float* backFrame, float2* frontFrame, float* output) {
    cudaStreamAttachMemAsync(NULL, backFrame, 0, cudaMemAttachGlobal);
    cudaStreamAttachMemAsync(NULL, frontFrame, 0, cudaMemAttachGlobal);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
    cudaStreamSynchronize(NULL);
	CudaPhase::resynthesis(output, backFrame, frontFrame, this->imp, this->nSamps, &this->ifft, this->hopSize);
	cudaStreamAttachMemAsync(NULL, backFrame, 0, cudaMemAttachHost);
	cudaStreamAttachMemAsync(NULL, frontFrame, 0, cudaMemAttachHost);
	cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachHost);
	cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}

   void PhaseVocoder::resynthesis(float* backFrame, float2* magFreq, float* output, void (*processing)()) {
   }