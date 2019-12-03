#include "RtAudio.h"
#include "kernel.h"
#include "cufft_.h"
#include "phaseVocoder.h"
#include "RtError.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#define RT
using namespace std;

float2* magFreq_2D[2048/64];
float2* output_2D[2048/64];
float final_output_[2048];
typedef struct {
} UserData;

float tranfer [256];
int callback(void *outputBuffer, void* inputBuffer, unsigned int nBufferFrames, double streamTime, RtAudioStreamStatus status, void *UserData) {
	float *h_inBuffer = (float *)inputBuffer; 
	float *h_outBuffer = (float *)outputBuffer;
	float *prevStream = (float *)UserData;
	PhaseVocoder* pv = (PhaseVocoder*)UserData;
	if (status) std::cout << "Stream underflow detected!" << std::endl;
    //memcpy(tranfer, inputBuffer, sizeof(float) * nBufferFrames);
	pv->analysis(h_inBuffer);
	pv->resynthesis(pv->prev_output, pv->curr_mag_phase, h_outBuffer);
    //memcpy(outputBuffer, inputBuffer, sizeof(float) * nBufferFrames * 2);
	return 0;
}

int main()
{
  int channels = 1;
  int sampleRate = 44100;
  PhaseVocoder* phase = new PhaseVocoder(256);
  unsigned int bufferSize = phase->nSamps;
  unsigned int bufferBytes = 256;
  int nBuffers = 4;
  int device = 0;
  RtAudio dac;
  RtAudio::StreamParameters input_params;
  RtAudio::StreamParameters output_params;
  unsigned int devices = dac.getDeviceCount();
  for(unsigned int i = 0; i < devices; i++){
    RtAudio::DeviceInfo info = dac.getDeviceInfo(i);
    if(info.probed == true)
      std::cout << "device" << i << " = "<< info.name << std::endl;
  }
  //input_params.deviceId = dac.getDefaultInputDevice();
  input_params.deviceId = dac.getDefaultInputDevice();
  input_params.nChannels = channels;
  output_params.deviceId = dac.getDefaultOutputDevice();
  output_params.nChannels = channels;
#ifdef RT
  try{
    //change buffer allocation to be cudamallocmanaged
    dac.openStream(&output_params, &input_params, RTAUDIO_FLOAT32,
                      sampleRate, &bufferSize, &callback, (void*)phase);
    dac.startStream();
  } 
  catch (RtError &error){
    error.printMessage();
    exit(EXIT_FAILURE);
  }
char cinput;
std::cout << "\n Recording in Progress: Do Not Eat\n";
std::cin.get(cinput);
  try{
    dac.stopStream();
  }
  catch(RtError &error){
    error.printMessage();
  }
  if(dac.isStreamOpen()) dac.closeStream();

#else // RT
  float input[2048];
  read_file("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/2048smp@44100.dat", input, 2048);

  save_results("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/input", input, phase->nSamps, 44100);
  cout << "premalloc" << endl;
  float* d_input;
  cudaMalloc((void**)&d_input, sizeof(float) * phase->nSamps);

  float2* outputtt;
  cudaMallocManaged((void**)&outputtt, sizeof(float2) * phase->nSamps, cudaMemAttachHost);
   checkCUDAError_("d_input memcpy backframe main.cpp final_output", __LINE__);
  //analysis
  for (int i = 0; i < phase->hopSize; i += phase->hopSize) {
	  float2* output, *magFreq;
	  cudaMallocManaged((void**)&output, sizeof(float2) * 2 * phase->nSamps);
      cudaMallocManaged((void**)&magFreq, sizeof(float2) * 2 * phase->nSamps);
	  cudaMemcpy(d_input, &input[i], sizeof(float)* phase->nSamps, cudaMemcpyHostToDevice);
   checkCUDAError_("input memcpy backframe main.cpp final_output", __LINE__);
	  cudaStreamSynchronize(NULL);
	  phase->analysis(d_input, output, magFreq);
   checkCUDAError_("analysis error", __LINE__);
	  output_2D[i / phase->hopSize] = output;
	  magFreq_2D[i / phase->hopSize] = magFreq;
	  cudaMemcpy(outputtt, output, sizeof(float2) * phase->nSamps, cudaMemcpyHostToHost);
   checkCUDAError_("output memcpxy anaged backframe main.cpp final_output", __LINE__);
	  printArray(phase->nSamps, outputtt);
  }

  // create emptyt backFrame
  float* backFrame;
  cudaMallocManaged((void**)&backFrame, sizeof(float) * phase->nSamps, cudaMemAttachHost);
   checkCUDAError_("mallloc managed backframe main.cpp final_output", __LINE__);
  for (int i = 0; i < phase->nSamps; i++) {
	  backFrame[i] = 0;
  }

  //resynthesis
  for (int i = 0; i < phase->hopSize; i += phase->hopSize) {
	  float *final_output;
	  cudaMallocManaged((void**)&final_output, sizeof(float2) * 2 * phase->nSamps);
	  checkCUDAError_("mallloc managed main.cpp final_output", __LINE__);
	  phase->resynthesis(backFrame, magFreq_2D[i/64], final_output);
	  cudaMemcpy(backFrame, final_output, sizeof(float) * phase->nSamps, cudaMemcpyHostToHost);
	  cout << "writing" << endl;
	  save_results("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/woop1", backFrame, 2 * phase->nSamps, 44100);
	  cudaFree(final_output);
  }
  cudaFree(backFrame);
#endif //RT
 
  for (int i = 0; i < 2048; i += phase->nSamps) {
		cudaFree(output_2D[i/phase->nSamps]);
		cudaFree(magFreq_2D[i/phase->nSamps]);
  }
  cudaFree(phase->imp);
  cufftDestroy(phase->plan);
  return 0;
}
