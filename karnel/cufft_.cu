#include <cuda.h>
#include <cuda_runtime.h>
#include "cufft_.h"
#include <chrono>
#include <cufft.h>

namespace FFT {
	namespace CuFFT {
		using FFT::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		/**
		 *Computes FFT using cuda library
		 */
		void computeCuFFT(float2* h_signal, int size) {
			timer().startGpuTimer();
			cudaStreamAttachMemAsync(NULL, h_signal, 0, cudaMemAttachGlobal);
			cufftHandle plan; 
			cufftPlan1d(&plan, size * sizeof(float2*), CUFFT_C2C, 1);
			cufftExecC2C(plan, (cufftComplex *)h_signal, (cufftComplex *)h_signal, CUFFT_FORWARD);
			timer().endGpuTimer();
		}
	}
}
