#include "phaseVocoder.h"
#include <cstdlib>
#include "io.h"
using namespace std;

void PhaseVocoder::analysis(float* input)
{
	checkCUDAError_("cudaMemcpy error", __LINE__);
//	cudaStreamAttachMemAsync(NULL, this->curr_input, 0, cudaMemAttachGlobal);
	//checkCUDAError_("attach input", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->curr_mag_phase, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach out", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach win", __LINE__);
	cudaStreamSynchronize(NULL);
	CudaPhase::pv_analysis(this->curr_mag_phase, this->curr_input, this->imp, this->nSamps, &this->plan);
   // cudaStreamAttachMemAsync(NULL, this->curr_input, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachHost);
	checkCUDAError_("attach imp", __LINE__);
	cudaStreamAttachMemAsync(NULL, this->curr_mag_phase, 0, cudaMemAttachHost);
	checkCUDAError_("reattach mag", __LINE__);
	cudaStreamSynchronize(NULL);
}
void PhaseVocoder::analysis()
{
	cudaStreamAttachMemAsync(NULL, this->curr_input, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach input", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->curr_mag_phase, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach out", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach win", __LINE__);
	cudaStreamSynchronize(NULL);
	CudaPhase::pv_analysis(this->curr_mag_phase, this->curr_input, this->imp, this->nSamps, &this->plan);
    cudaStreamAttachMemAsync(NULL, this->curr_input, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachHost);
	cudaStreamAttachMemAsync(NULL, this->curr_mag_phase, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}
void PhaseVocoder::analysis(float* input, float2* output, int offset){
	printf("in the cut\n");
    cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach out", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach win", __LINE__);
	cudaStreamSynchronize(NULL);
	checkCUDAError_("stream sync error pre anal", __LINE__);
	CudaPhase::pv_analysis(output, input, this->imp, this->nSamps, &this->plan, offset);
	cudaStreamSynchronize(NULL);
	checkCUDAError_("stream sync error post anal", __LINE__);
	printf("in the cut\n");
	checkCUDAError_("pv_analysis ", __LINE__);
    cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachHost);
	checkCUDAError_("attach out", __LINE__);
}

void PhaseVocoder::analysis(float* input, float2* output){
	printf("in the cut\n");
    cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach out", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach win", __LINE__);
	cudaStreamSynchronize(NULL);
	CudaPhase::pv_analysis(output, input, this->imp, this->nSamps, &this->plan);
	cudaStreamSynchronize(NULL);
	checkCUDAError_("stream sync error", __LINE__);
	printf("in the cut\n");
	checkCUDAError_("pv_analysis ", __LINE__);
    cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachHost);
	checkCUDAError_("attach out", __LINE__);
}
void PhaseVocoder::analysis(float* input, float2* output, float2* magFreq){
	cudaStreamAttachMemAsync(NULL, input, 0, cudaMemAttachGlobal);
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
	
   checkCUDAError_("memcpy backframe error", __LINE__);
    cudaStreamAttachMemAsync(NULL, frontFrame, 0, cudaMemAttachGlobal);
   checkCUDAError_("memcpy backframe error", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->imp1, 0, cudaMemAttachGlobal);
   checkCUDAError_("memcpy backframe error", __LINE__);
    cudaStreamSynchronize(NULL);
	CudaPhase::resynthesis(output, backFrame, frontFrame, this->imp1, this->nSamps, &this->ifft, this->hopSize);
	cudaStreamAttachMemAsync(NULL, backFrame, 0, cudaMemAttachHost);
	cudaStreamAttachMemAsync(NULL, frontFrame, 0, cudaMemAttachHost);
	cudaStreamAttachMemAsync(NULL, this->imp1, 0, cudaMemAttachHost);
	cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}

   void PhaseVocoder::resynthesis(float* backFrame, float2* magFreq, float* output, void (*processing)()) {
   }