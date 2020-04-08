/* Nikil's streamlined version
 * Removing extra test cases and adding comments so I know what the hell is
 * going on.
 */

/* includes */
#include "RtAudio.h"
#include "AudioFile.h"
#include "RtError.h"
#include "cufft_.h"
#include "io.h"
#include "kernel.h"
#include "phaseVocoder.h"
#include "testing_helpers.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

/* defines */
//#define OVERLAPTEST
//#define DEBUG
//#define DEBUGRESYNTH
//#define ANALYSIS_PERFORMANCE_ANALYSIS
//#define RESYNTHESIS_PERFORMANCE_ANALYSIS
#define TWOD
#define USECUFFTRE
#define USECUFFTAN
//#define RT

/* namespace */
using namespace std;

void bufferToManageMemory(AudioFile<float> *buffer, float *output,
                          int channel) {
  cudaMallocManaged((void **)&output,
                    buffer->getNumSamplesPerChannel() * sizeof(float),
                    cudaMemAttachHost);
  checkCUDAError_("Malloc Error: unified memory for samples", __LINE__);
  /*cudaMemcpy(output, buffer.samples[channel], buffer.getNumSamplesPerChannel()
  * sizeof(float),cudaMemcpyHostToHost ); checkCUDAError_("Memcpy Error: unified
  memory for samples", __LINE__);
  */
  for (int i = 0; i < buffer->getNumSamplesPerChannel(); i++) {
    output[i] = buffer->samples[channel][i];

    checkCUDAError_("mem orr", __LINE__);
  }
}
#ifdef RT
float tranfer[256];
int callback(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
             double streamTime, RtAudioStreamStatus status, void *UserData) {
  float *h_inBuffer = (float *)inputBuffer;
  float *h_outBuffer = (float *)outputBuffer;
  float *prevStream = (float *)UserData;
  PhaseVocoder *pv = (PhaseVocoder *)UserData;
  if (status)
    std::cout << "Stream underflow detected!" << std::endl;
  memcpy(pv->curr_input, inputBuffer, sizeof(float) * nBufferFrames);
  pv->analysis();
  //	pv->resynthesis(pv->prev_output, pv->curr_mag_phase, h_outBuffer);
  // memcpy(outputBuffer, inputBuffer, sizeof(float) * nBufferFrames * 2);
  return 0;
}
#endif
int main(int argc, char **argv) {
  std::string file = "/testtones/";
  std::string outfile = "/output/";
  Effect effect;
  if (argc < 2) {
    file = "/testtones/1000sine.wav";
    effect = Effect::TIME_SHIFT;
    outfile.append("out.wav");
  } else if (argc < 3) {
    file.append(argv[1]);
    effect = Effect::TIME_SHIFT;
    outfile.append("out.wav");
  } else if (argc < 4) {
    file.append(argv[1]);
    effect = static_cast<Effect>(*argv[2]);
    outfile.append("out.wav");
  } else if (argc < 5) {
    file.append(argv[1]);
    effect = static_cast<Effect>(*argv[2]);
    outfile.append(argv[3]);
  }
  int channels = 1;
  int sampleRate = 44100;
  PhaseVocoder *phase = new PhaseVocoder(256, effect, 1, 2);
  unsigned int bufferSize = phase->nSamps;
  unsigned int bufferBytes = 256;
  int nBuffers = 4;
  int device = 0;
  RtAudio dac;
  RtAudio::StreamParameters input_params;
  RtAudio::StreamParameters output_params;
  unsigned int devices = dac.getDeviceCount();
  for (unsigned int i = 0; i < devices; i++) {
    RtAudio::DeviceInfo info = dac.getDeviceInfo(i);
    if (info.probed == true)
      std::cout << "device" << i << " = " << info.name << std::endl;
  }
  // input_params.deviceId = dac.getDefaultInputDevice();
  input_params.deviceId = 11;
  input_params.nChannels = channels;
  output_params.deviceId = 11;
  output_params.nChannels = channels;
#ifdef RT
  try {
    // change buffer allocation to be cudamallocmanaged
    dac.openStream(&output_params, &input_params, RTAUDIO_FLOAT32, sampleRate,
                   &bufferSize, &callback, (void *)phase);
    dac.startStream();
  } catch (RtError &error) {
    error.printMessage();
    exit(EXIT_FAILURE);
  }
  char cinput;
  std::cout << "\n Recording in Progress: Do Not Eat\n";
  std::cin.get(cinput);
  try {
    dac.stopStream();
  } catch (RtError &error) {
    error.printMessage();
  }
  if (dac.isStreamOpen())
    dac.closeStream();

#else // RT
  cout << "Offline Vocoding" << endl;

  AudioFile<float> *audioFile = new AudioFile<float>();
  AudioFile<float> outFile;
  if (!audioFile->load(file)) {
    cout << "err: wav failed to load" << endl;
    exit(1);
  }
  audioFile->printSummary();
  int numChannels = audioFile->getNumChannels();
  int numSamples = audioFile->getNumSamplesPerChannel();

  // outFile.setAudioBufferSize(numChannels, phase->timeScale * numSamples);
  outFile.setAudioBufferSize(2, phase->timeScale * numSamples);

  outFile.setSampleRate(44100);
  outFile.setBitDepth(16);
#ifdef TWOD
  float **d_input;
  cout << "Loading file...." << endl;
  cudaMallocManaged((void **)&d_input, sizeof(float *) * numChannels,
                    cudaMemAttachHost);
  checkCUDAError_("Error creating unified memory for channels", __LINE__);
  for (int i = 0; i < numChannels; i++) {
    cudaMallocManaged((void **)&d_input[i],
                      audioFile->getNumSamplesPerChannel() * sizeof(float),
                      cudaMemAttachHost);
    checkCUDAError_("Malloc Error: unified memory for samples", __LINE__);
    for (int j = 0; j < audioFile->getNumSamplesPerChannel(); j++) {
      d_input[i][j] = audioFile->samples[i][j];
    }
  }
#ifdef OVERLAPTEST
  float *intermediary1;
  cudaMalloc((void **)&intermediary1, sizeof(float) * phase->nSamps);
  checkCUDAError_("Error Mallocing Intermediary in main.cpp", __LINE__);
  float *backFrame1;
  cudaMallocManaged((void **)&backFrame1, sizeof(float) * phase->nSamps,
                    cudaMemAttachHost);
  checkCUDAError_("mallloc managed backframe main.cpp", __LINE__);
  for (int i = 0; i < phase->nSamps; i++) {
    backFrame1[i] = 0;
  }
  cout << "overlap test" << endl;
  for (int j = 0; j < numChannels; j++) {
    int outIndex = 0;
    cudaStreamAttachMemAsync(NULL, d_input[j], 0, cudaMemAttachGlobal);
    cudaStreamAttachMemAsync(NULL, backFrame1, 0, cudaMemAttachGlobal);
    checkCUDAError_("stream attach", __LINE__);
    for (int i = 0; i < numSamples / phase->hopSize; i++) {
      cudaStreamAttachMemAsync(NULL, backFrame1, 0, cudaMemAttachGlobal);
      float *final_output;
      cudaMallocManaged((void **)&final_output, sizeof(float) * phase->nSamps,
                        cudaMemAttachGlobal);
      checkCUDAError_("malloc", __LINE__);
      phase->test_overlap_add(&d_input[j][i * phase->hopSize], final_output,
                              intermediary1, backFrame1, 256);
      cudaMemcpy(backFrame1, final_output, sizeof(float) * phase->nSamps,
                 cudaMemcpyDeviceToDevice);
      checkCUDAError_("memcpy", __LINE__);
      cudaStreamAttachMemAsync(NULL, backFrame1, 0, cudaMemAttachHost);
      cudaStreamSynchronize(NULL);
      for (int a = 0; a < phase->outHopSize; a++) {
        // int idx = i /phase->hopSize * phase->outHopSize + j;
        int idx = outIndex + a;
        if (idx < 0) {
          break;
        }
        outFile.samples[j][idx] = backFrame1[a];
        if (numChannels = 1) {
          outFile.samples[1][idx] = backFrame1[a];
        }
#ifdef DEBUGRESYNTH
        printf("%f\n", backFrame[j]);
#endif
      }
      outIndex += phase->outHopSize;
      cudaFree(final_output);
      checkCUDAError_("cuda", __LINE__);
    }
  }
  goto write;
#endif

  float2 ***d_output;
  cudaMallocManaged((void **)&d_output, sizeof(float2 **) * numChannels,
                    cudaMemAttachHost);
  checkCUDAError_("Error creating unified memory for channel output", __LINE__);
  float2 *zeros;
  cudaMalloc((void **)&zeros, sizeof(float2) * 2 * phase->nSamps);
  checkCUDAError_("Error creating zero array", __LINE__);
  for (int i = 0; i < numChannels; i++) {
    cudaMallocManaged((void **)&d_output[i],
                      sizeof(float2 *) * numSamples / phase->hopSize,
                      cudaMemAttachHost);
    checkCUDAError_("Erroring Mallocing 2D_analysis array", __LINE__);
    for (int j = 0; j < numSamples; j += phase->hopSize) {
      // cudaMallocManaged((void**)&d_output[i][j], sizeof(float2) *
      // phase->nSamps, cudaMemAttachHost);
      cudaMalloc((void **)&d_output[i][j / phase->hopSize],
                 sizeof(float2) * 2 * phase->nSamps);
      cudaMemcpy(d_output[i][j / phase->hopSize], zeros,
                 sizeof(float2) * 2 * phase->nSamps, cudaMemcpyDeviceToDevice);
      checkCUDAError_("Erroring Mallocing analysis sample array", __LINE__);
    }
  }
  cout << "analysis..." << endl;
  // analysis
  float *intermediary;
  cudaMalloc((void **)&intermediary, sizeof(float) * phase->nSamps);
  checkCUDAError_("Error Mallocing Intermediary in main.cpp", __LINE__);
  float2 *fft;
  cudaMallocManaged((void **)&fft, sizeof(float2) * 2 * phase->nSamps,
                    cudaMemAttachGlobal);
  checkCUDAError_("Error Mallocing fft in main.cpp", __LINE__);
  for (int channel = 0; channel < numChannels; channel++) {
    cudaStreamAttachMemAsync(NULL, d_input[channel], 0, cudaMemAttachGlobal);
    checkCUDAError_("attach input", __LINE__);
    for (int i = 0; i < numSamples - phase->hopSize; i += phase->hopSize) {
      cudaStreamSynchronize(NULL);
#ifdef USECUFFTAN
      phase->analysis_CUFFT(&d_input[channel][i],
                            d_output[channel][i / phase->hopSize], fft,
                            intermediary);
#else
      phase->analysis(&d_input[channel][i],
                      d_output[channel][i / phase->hopSize], fft, intermediary);
#endif
#ifdef ANALYSIS_PERFORMANCE_ANALYSIS
      // printElapsedTime(CudaPhase::timer().getGpuElapsedTimeForPreviousOperation(),
      // "Analysis");
      cout << CudaPhase::timer().getGpuElapsedTimeForPreviousOperation()
           << endl;
#endif
#ifdef DEBUG
      float2 *debug_arr;
      cudaMallocManaged((void **)&debug_arr, sizeof(float2) * 2 * phase->nSamps,
                        cudaMemAttachHost);
      cudaMemcpy(debug_arr, d_output[channel][i / phase->hopSize],
                 sizeof(float2) * 2 * phase->nSamps, cudaMemcpyDeviceToHost);
      printArraywNewLines(2 * phase->nSamps, debug_arr);
#endif
      checkCUDAError_("analysis error main.cpp", __LINE__);
    }
  }
  cudaFree(intermediary);
  // create empty backFrame
  float *backFrame;
  cudaMallocManaged((void **)&backFrame, sizeof(float) * phase->nSamps,
                    cudaMemAttachHost);
  checkCUDAError_("mallloc managed backframe main.cpp", __LINE__);
  for (int i = 0; i < phase->nSamps; i++) {
    backFrame[i] = 0;
  }
  cout << "resynthesis..." << endl;
  // resynthesis
  switch (effect) {
    // TIMESHIFT
  case Effect::TIME_SHIFT: {
    for (int channel = 0; channel < numChannels; channel++) {
      int outIndex = 0;
      for (int i = 0; i < numSamples / phase->outHopSize; i++) {
        float *final_output;
        cudaMallocManaged((void **)&final_output, sizeof(float) * phase->nSamps,
                          cudaMemAttachHost);
        checkCUDAError_("mallloc managed main.cpp final_output", __LINE__);
#ifdef USECUFFTRE
        phase->resynthesis_CUFFT(backFrame, d_output[channel][i], final_output);
#else
        phase->resynthesis(backFrame, d_output[channel][i], fft, final_output);
#endif
#ifdef RESYNTHESIS_PERFORMANCE_ANALYSIS
        // printElapsedTime(CudaPhase::timer().getGpuElapsedTimeForPreviousOperation(),
        // "Resynthesis");
        cout << CudaPhase::timer().getGpuElapsedTimeForPreviousOperation()
             << endl;
#endif
        cudaMemcpy(backFrame, final_output, sizeof(float) * phase->nSamps,
                   cudaMemcpyHostToHost);
        cudaFree(final_output);
        for (int j = 0; j < phase->outHopSize; j++) {
          // int idx = i /phase->hopSize * phase->outHopSize + j;
          int idx = outIndex + j;
          if (idx < 0) {
            break;
          }
          outFile.samples[channel][idx] = backFrame[j];
          if (numChannels = 1) {
            outFile.samples[1][idx] = backFrame[j];
          }
#ifdef DEBUGRESYNTH
          printf("%f\n", backFrame[j]);
#endif
        }
        outIndex += phase->outHopSize;
      }
    }
    break;
  }
  // pitchSHIFT
  case Effect::PITCH_SHIFT: {
  }
  }

  cudaFree(fft);
write:
  cout << "writing to file" << endl;
  outFile.save("/output/out.wav");
  cudaFree(backFrame);
  for (int i = 0; i < numChannels; i++) {
    for (int j = 0; j < numSamples / phase->hopSize; j++) {
      cudaFree(d_output[i][j]);
    }
    cudaFree(d_input[i]);
    cudaFree(d_output[i]);
  }
#else
  float *d_input;
  numChannels = 1;
  cudaMallocManaged((void **)&d_input,
                    audioFile->getNumSamplesPerChannel() * sizeof(float),
                    cudaMemAttachHost);
  checkCUDAError_("Malloc Error: unified memory for samples", __LINE__);
  for (int i = 0; i < audioFile->getNumSamplesPerChannel(); i++) {
    d_input[i] = audioFile->samples[0][i];
  }
  // save_results("/output.out", d_input, numSamples, 44100);
  checkCUDAError_("mem orr", __LINE__);
  // bufferToManageMemory(audioFile, d_input, 0);
  delete audioFile;
  cout << "mallocing for output" << endl;
  float2 **d_output;
  cudaMallocManaged((void **)&d_output,
                    sizeof(float2 *) * numSamples / phase->hopSize,
                    cudaMemAttachHost);
  checkCUDAError_("Error creating unified memory for channel output", __LINE__);
  for (int j = 0; j < numSamples; j += phase->hopSize) {
    cudaMalloc((void **)&d_output[j / phase->hopSize],
               sizeof(float2) * phase->nSamps);
    checkCUDAError_("Erroring Mallocing analysis sample array", __LINE__);
  }
  cout << "analysis..." << endl;
  // analysis
  cudaStreamAttachMemAsync(NULL, d_input, 0, cudaMemAttachGlobal);
  checkCUDAError_("malloc ini", __LINE__);
  for (int i = 0; i < numSamples - phase->hopSize; i += phase->hopSize) {
    cudaStreamSynchronize(NULL);
    phase->analysis(d_input, d_output[i / phase->hopSize], i);
    checkCUDAError_("analysis error main.cpp", __LINE__);
#ifdef DEBUG
    float2 *debug_arr;
    cudaMallocManaged((void **)&debug_arr, sizeof(float2) * phase->nSamps,
                      cudaMemAttachHost);
    cudaMemcpy(debug_arr, d_output[i / phase->hopSize],
               sizeof(float2) * phase->nSamps, cudaMemcpyDeviceToHost);
    printArraywNewLines(phase->nSamps, debug_arr);
#endif
  }
  cout << "analysis over" << endl;
  cudaFree(d_input);
  checkCUDAError_("freeing input", __LINE__);
  // create empty backFrame
  float *backFrame;
  cudaMallocManaged((void **)&backFrame, sizeof(float) * phase->nSamps,
                    cudaMemAttachHost);
  checkCUDAError_("malloc managed backframe main.cpp", __LINE__);
  for (int i = 0; i < phase->nSamps; i++) {
    backFrame[i] = 0;
  }
  cout << "resynthesis" << endl;
  // resynthesis
  for (int i = 0; i < numSamples - phase->hopSize; i += phase->hopSize) {
    float *final_output;
    cudaMallocManaged((void **)&final_output, sizeof(float) * phase->nSamps,
                      cudaMemAttachHost);
    checkCUDAError_("mallloc managed main.cpp final_output", __LINE__);
    phase->resynthesis(backFrame, d_output[i / phase->hopSize], final_output);
    cudaMemcpy(backFrame, final_output, sizeof(float) * phase->nSamps,
               cudaMemcpyHostToHost);
    checkCUDAError_("Memcpy output to backframe", __LINE__);
    cudaFree(final_output);
    checkCUDAError_("cudaFree final output", __LINE__);
    printArray(phase->nSamps, backFrame);
    for (int j = 0; j < phase->outHopSize; j++) {
      int idx = i / phase->hopSize * phase->outHopSize + j;
      cout << idx << endl;
      outFile.samples[0][idx] = backFrame[j];
      outFile.samples[1][idx] = backFrame[j];
    }
  }
  cout << "writing to file" << endl;
  outFile.save("/output/out.wav");
  cudaFree(backFrame);
  for (int i = 0; i < numChannels; i++) {
    for (int j = 0; j < numSamples / phase->hopSize; j++) {
      cudaFree(d_output[j]);
    }
    cudaFree(d_input);
    cudaFree(d_output);
  }
#endif
#endif // RT

  cudaFree(phase->imp);
  cufftDestroy(phase->plan);
  return 0;
}