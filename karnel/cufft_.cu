#include <cuda.h>
#include <cuda_runtime.h>
#include "cufft_.h"
#include <chrono>
#include "cufft.h"

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
			float2* d_signal;
			cudaMalloc((void**)d_signal, sizeof(float2*) * size);
			cudaMemcpy(d_signal, h_signal, sizeof(float2*) * size, cudaMemcpyHostToDevice);
			timer().startGpuTimer();
			cufftHandle plan; 
			cufftPlan1d(&plan, size * sizeof(float2*), CUFFT_C2C, 1);

			cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);
			timer().endGpuTimer();
			cudaMemcpy(h_signal, d_signal, sizeof(float2*) * size, cudaMemcpyDeviceToHost);

			free(d_signal);
		}
	}
}
