#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>


namespace Additive {

	void initSynth(int numSinusoids, int numSamples, float* host_frequencies);
    void compute_sinusoid_gpu_simple(float * buffer, float angle);
    void endSynth();
	void initSynth_THX(int numSinusoid, int numSample, float* host_start_freq, float* host_end_freq,
		float* host_angle, float*  host_gains, float slide);
	void compute_sinusoid_hybrid(float* buffer, float * time);
	void endSynth_THX();
	
}
