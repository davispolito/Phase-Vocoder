#pragma once

#include "common.h"
#include "kernel.h"

#define NUM_STREAMS 3

enum Effect { TIME_SHIFT = 't', PITCH_SHIFT = 'p' };

class PhaseVocoder {
private:
  const int sampling_rate = 44100;

  // Number of groups of N samples
  int nGroups;

  // I have no idea what this does
  const float beta = 6.8f;

public:
  float *imp;
  float *imp1;
  float *curr_input;
  float *prev_input;
  float *prev_output;
  float2 *prev_mag_phase;
  float2 *curr_mag_phase;
  cufftHandle plan;
  cufftHandle ifft;

  int hopSize;

  int nSamps; // nSamps  - Length of original signal
  int R = 1;  // R - Compression Ratio
  int N;      // N - num Phase Vocoder Channels

  float timeScale;
  int outHopSize;
  int stream;
  cudaStream_t streams[NUM_STREAMS];

  void checkCUDAErrori(const char *msg, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      if (line >= -1) {
        fprintf(stderr, "Line %d: ", line);
      }
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  PhaseVocoder(int samples)
      : nSamps(samples), hopSize(samples / 2), timeScale(1) {
    cudaMallocManaged((void **)&imp, sizeof(float) * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc imp error", __LINE__);
    cudaMallocManaged((void **)&imp1, sizeof(float) * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc imp1 error", __LINE__);
    cudaMallocManaged((void **)&prev_mag_phase, sizeof(float) * 2 * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc prev_mag_phase error", __LINE__);
    cudaMallocManaged((void **)&prev_input, sizeof(float) * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc prev_input", __LINE__);
    cudaMallocManaged((void **)&prev_output, sizeof(float) * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc prev_output", __LINE__);
    cudaMallocManaged((void **)&curr_mag_phase, sizeof(float) * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc curr_mag_phase", __LINE__);
    // cudaMallocManaged((void**)&curr_input, sizeof(float) * samples,
    // cudaMemAttachHost);
    cudaMallocManaged((void **)&curr_input, sizeof(float) * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc curr_input", __LINE__);
    // hanning window
    // hanning window
    for (int i = 0; i < samples; i++) {
      imp[i] = 0.5f * (1.f - cosf(2.f * M_PI * i / samples));
    }
    for (int i = 0; i < 2 * samples; i++) {
      imp1[i] = 0.5f * (1.f - cosf(2.f * M_PI * i / samples));
    }
    cufftPlan1d(&(this->plan), 2 * samples, CUFFT_C2C, 1);
    checkCUDAErrori("Cufft Plan FFT Error", __LINE__);
    cufftPlan1d(&(this->ifft), 2 * samples, CUFFT_C2R, 1);
    checkCUDAErrori("Cufft Plan IFFT Error", __LINE__);
    outHopSize = hopSize * timeScale;
    for (int i = 0; i < NUM_STREAMS; i++) {
      cudaStreamCreate(&streams[i]);
    }
  }
  PhaseVocoder(int samples, Effect e, float scaleFactor, int hop)
      : nSamps(samples), hopSize(samples / hop) {
    cudaMallocManaged((void **)&imp, sizeof(float) * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc imp error", __LINE__);
    cudaMallocManaged((void **)&imp1, sizeof(float) * samples,
                      cudaMemAttachHost);
    checkCUDAErrori("Malloc imp1 error", __LINE__);
    // hanning window
    float omega = 2.f * M_PI / (samples - 1);
    for (int i = 0; i < samples; i++) {
      //  imp[i] = 0.5f * (1.f - cosf(omega * i));
      imp[i] = (0.54f - 0.46f * cos(omega * (i)));
    }
    float omega1 = 2.f * M_PI / (2 * samples - 1);
    for (int i = 0; i < samples; i++) {
      // imp[i] = 0.5f * (1.f - cosf(omega * i));
      imp1[i] = (0.54f - 0.46f * cos(omega1 * (i)));
    }
    cufftPlan1d(&(this->plan), 2 * samples, CUFFT_C2C, 1);
    checkCUDAErrori("Cufft Plan FFT Error", __LINE__);
    cufftPlan1d(&(this->ifft), 2 * samples, CUFFT_C2R, 1);
    checkCUDAErrori("Cufft Plan IFFT Error", __LINE__);

    switch (e) {
    case TIME_SHIFT: {
      timeScale = scaleFactor;
      outHopSize = scaleFactor * hopSize;
      break;
    }
    case PITCH_SHIFT: {

      break;
    }
    }
    for (int i = 0; i < NUM_STREAMS; i++) {
      cudaStreamCreate(&streams[i]);
    }
  }

  cudaStream_t *getStream() {
    cudaStream_t *out = &streams[stream++];
    stream %= NUM_STREAMS;
    return out;
  }

  cudaStream_t *getPrevStream() { return &streams[(stream - 1) % NUM_STREAMS]; }

  ~PhaseVocoder() { cudaFree(imp); }
  void analysis();
  void analysis(float *input);
  void analysis(float *input, float2 *output, float2 *fft, float *intermediary);
  void analysis_CUFFT(float *input, float2 *output, float2 *fft,
                      float *intermediary);
  void resynthesis(float *backFrame, float2 *frontFrame, float2 *intermediary,
                   float *output);
  void resynthesis(float *backFrame, float2 *frontFrame, float *output,
                   void (*processing)());
  void test_overlap_add(float *input, float *output, float *intermediary,
                        float *backFrame, int N);
  void resynthesis_CUFFT(float *backFrame, float2 *frontFrame, float *output);
};