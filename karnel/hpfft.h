#pragma once 
#include "common.h"
namespace FFT {
	namespace HPFFT {
		FFT::Common::PerformanceTimer& timer();
		float2* computeFFTSh(float2* h_signal, int size, int radix, int numThreads);
		float2* computeFFTCooley(float2* h_signal, int N, int R, int numThreads);
		void computeGPUFFT(int N, int R, float2* h_signal, float2* intermediary);
		void computeGPUFFT_RT(int N, int R, float2* h_signal, float2* intermediary, cudaStream_t* stream);
		void computeGPUIFFT(int N, int R, float2* h_signal, float2* intermediary);
		void computeGPUIFFT_RT(int N, int R, float2* h_signal, float2* intermediary, cudaStream_t* stream);
		void FFT2(float2 * v);
	}
}
