#include <cuda.h>
#include <cuda_runtime.h>
#include "cufft.h"

namespace FFT {
	namespace CuFFT {
	FFT::Common::PerformanceTimer& timer();
	void computeCuFFT(float2* h_signal, int size);
	}
}
