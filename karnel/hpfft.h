#pragma once 
#include "common.h"
namespace FFT {
	namespace HPFFT {
		FFT::Common::PerformanceTimer& timer();
		float2* computeFFTSh(float2* h_signal, int size, int radix, int numThreads);
		float2* computeFFTCooley(float2* h_signal, int N, int R, int numThreads);
		void FFT2(float2 * v);
		float2* computeGPUFFT(int N, int R, float2* h_signal) ;
	}
}