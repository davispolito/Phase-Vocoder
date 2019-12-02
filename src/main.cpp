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
using namespace std;

float2* magFreq_2D[2048/64];
float2* output_2D[2048/64];
float final_output_[2048];

int main()
{
  float input[2048];

  int channels = 1;
  int sampleRate = 44100;
  unsigned int bufferSize, bufferBytes = 256;
  int nBuffers = 4;
  int device = 0;
  RtAudio dac;
  RtAudio::StreamParameters input_params;
  RtAudio::StreamParameters output_params;
  unsigned int devices = dac.getDeviceCount();
  for(int i = 0; i < devices; i++){
    RtAudio::DeviceInfo info = dac.getDeviceInfo(i);
    if(info.probed == true)
      std::cout << "device" << i << " = "<< info.name << std::endl;
  }
  //input_params.deviceId = dac.getDefaultInputDevice();
  input_params.deviceId = dac.getDefaultInputDevice();
  input_params.nChannels = 2;
  output_params.deviceId = dac.getDefaultOutputDevice();
  output_params.nChannels = 2;
  PhaseVocoder* phase = new PhaseVocoder(256);
#ifdef RT
  try{
    //change buffer allocation to be cudamallocmanaged
    dac.openStream(&output_params, &input_params, RTAUDIO_SINT16,
                      sampleRate, &bufferSize, &callback, (void*)&bufferBytes);
    bufferBytes = bufferSize * 2 * 2;
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
  read_file("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/2048smp@44100.dat", input, 2048);

  save_results("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/input", input, phase->nSamps, 44100);
  cout << "premalloc" << endl;
  float* d_input;
  cudaMalloc((void**)&d_input, sizeof(float) * phase->nSamps);

  //analysis
  for (int i = 0; i < phase->hopSize; i += phase->hopSize) {
	  float2* output, *magFreq;
	  cudaMallocManaged((void**)&output, sizeof(float2) * 2 * phase->nSamps);
      cudaMallocManaged((void**)&magFreq, sizeof(float2) * 2 * phase->nSamps);
	  cudaMemcpy(d_input, &input[i], sizeof(float)* phase->nSamps, cudaMemcpyHostToDevice);
	  cudaStreamSynchronize(NULL);
	  phase->analysis(d_input, output, magFreq);
	  output_2D[i / phase->hopSize] = output;
	  magFreq_2D[i / phase->hopSize] = magFreq;
  }

  // create emptyt backFrame
  float* backFrame;
  cudaMallocManaged((void**)&backFrame, sizeof(float) * phase->nSamps, cudaMemAttachHost);
  for (int i = 0; i < phase->nSamps; i++) {
	  backFrame[i] = 0;
  }

  //resynthesis
  for (int i = 0; i < phase->hopSize; i += phase->hopSize) {
	  float *final_output;
	  cudaMallocManaged((void**)&final_output, sizeof(float2) * 2 * phase->nSamps);
	  phase->resynthesis(backFrame, magFreq_2D[i/64], final_output);
	  cudaMemcpy(backFrame, final_output, sizeof(float) * phase->nSamps, cudaMemcpyHostToHost);
	  cout << "writing" << endl;
	  save_results("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/woop", backFrame, phase->nSamps, 44100);
	  cudaFree(final_output);
  }
#endif //RT
 
  for (int i = 0; i < 2048; i += phase->nSamps) {
		cudaFree(output_2D[i/phase->nSamps]);
		cudaFree(magFreq_2D[i/phase->nSamps]);
  }
  cudaFree(backFrame);
  cudaFree(phase->imp);
  cufftDestroy(phase->plan);
  return 0;
}
