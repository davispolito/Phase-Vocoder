#include "phaseVocoder.h"
#include <cstdlib>
#include "io.h"
using namespace std;

/*void PhaseVocoder::analysis()
{
	int idx = this->stream;
	cudaStream_t* stream = this->getStream();
	cudaStreamAttachMemAsync(NULL, this->input, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach input", __LINE__);
    cudaStreamAttachMemAsync(*stream, this->mag_phase[idx], 0);
	checkCUDAError_("attach magphase", __LINE__);
    cudaStreamAttachMemAsync(*stream, this->fft[idx], 0);
	checkCUDAError_("attach fft", __LINE__);
	//cudaStreamSynchronize(*stream);
    void pv_analysis_RT(float2* output,float2* fft, float* input, float * intermediary, int N, stream);
	//cudaStreamSynchronize(NULL);
}*/
void PhaseVocoder::test_overlap_add(float* input, float* output, float* intermediary, float* backFrame, int N){
	CudaPhase::test_overlap_add(input, output, intermediary, backFrame, this->imp, N, this->hopSize);
	cudaStreamSynchronize(NULL);
}

void PhaseVocoder::analysis_CUFFT(float* input, float2* output, float2* fft, float* intermediary){
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach win", __LINE__);
	cudaStreamSynchronize(NULL);
	CudaPhase::pv_analysis_CUFFT(output, fft, input, intermediary, this->imp, this->nSamps);
	cudaStreamSynchronize(NULL);
	checkCUDAError_("stream sync error", __LINE__);
	checkCUDAError_("pv_analysis ", __LINE__);
}
void PhaseVocoder::analysis(float* input, float2* output, float2* fft, float* intermediary){
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
	checkCUDAError_("attach win", __LINE__);
	cudaStreamSynchronize(NULL);
	CudaPhase::pv_analysis(output, fft, input, intermediary, this->imp, this->nSamps);
	cudaStreamSynchronize(NULL);
	checkCUDAError_("stream sync error", __LINE__);
	checkCUDAError_("pv_analysis ", __LINE__);
}

void PhaseVocoder::resynthesis(float* backFrame, float2* frontFrame, float2* intermediary, float* output) {
    cudaStreamAttachMemAsync(NULL, backFrame, 0, cudaMemAttachGlobal);
    checkCUDAError_("Attach Global Error backframe ", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
    checkCUDAError_("Attach Global error window", __LINE__);
    cudaStreamSynchronize(NULL);
	CudaPhase::resynthesis(output, backFrame, frontFrame, intermediary, this->imp, this->nSamps, this->outHopSize);
	cudaStreamAttachMemAsync(NULL, backFrame, 0, cudaMemAttachHost);
    checkCUDAError_("Attach Host error backFrame", __LINE__);
	cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachHost);
    checkCUDAError_("Attach Host error window", __LINE__);
	cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachHost);
    checkCUDAError_("Attach Host error output", __LINE__);
	cudaStreamSynchronize(NULL);
}

void PhaseVocoder::resynthesis_CUFFT(float* backFrame, float2* frontFrame, float* output) {
    cudaStreamAttachMemAsync(NULL, backFrame, 0, cudaMemAttachGlobal);
    checkCUDAError_("Attach Global Error backframe ", __LINE__);
    cudaStreamAttachMemAsync(NULL, this->imp, 0, cudaMemAttachGlobal);
    checkCUDAError_("Attach Global error window", __LINE__);
    cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachGlobal);
    checkCUDAError_("Attach Global error output", __LINE__);
    cudaStreamSynchronize(NULL);
	CudaPhase::resynthesis_CUFFT(output, backFrame, frontFrame, this->imp, this->nSamps, this->outHopSize);
	cudaStreamAttachMemAsync(NULL, backFrame, 0, cudaMemAttachHost);
    checkCUDAError_("Attach Host error backFrame", __LINE__);
	cudaStreamAttachMemAsync(NULL, this->imp1, 0, cudaMemAttachHost);
    checkCUDAError_("Attach Host error window", __LINE__);
	cudaStreamAttachMemAsync(NULL, output, 0, cudaMemAttachHost);
    checkCUDAError_("Attach Host error output", __LINE__);
	cudaStreamSynchronize(NULL);
}
   void PhaseVocoder::resynthesis(float* backFrame, float2* magFreq, float* output, void (*processing)()) {
   }