#include "RtAudio.h"
//#define DEBUG
#define TWOD
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
#include "AudioFile.h"
//#define RT
using namespace std;


void bufferToManageMemory(AudioFile<float> *buffer, float* output, int channel){
    cudaMallocManaged((void**)&output, buffer->getNumSamplesPerChannel() * sizeof(float), cudaMemAttachHost);
    checkCUDAError_("Malloc Error: unified memory for samples", __LINE__);
    /*cudaMemcpy(output, buffer.samples[channel], buffer.getNumSamplesPerChannel() * sizeof(float),cudaMemcpyHostToHost );
    checkCUDAError_("Memcpy Error: unified memory for samples", __LINE__);
    */
    for (int i = 0; i < buffer->getNumSamplesPerChannel(); i++){
      output[i] = buffer->samples[channel][i];
      
    checkCUDAError_("mem orr", __LINE__);
    }
}
float tranfer [256];
int callback(void *outputBuffer, void* inputBuffer, unsigned int nBufferFrames, double streamTime, RtAudioStreamStatus status, void *UserData) {
	float *h_inBuffer = (float *)inputBuffer; 
	float *h_outBuffer = (float *)outputBuffer;
	float *prevStream = (float *)UserData;
	PhaseVocoder* pv = (PhaseVocoder*)UserData;
	if (status) std::cout << "Stream underflow detected!" << std::endl;
  memcpy(pv->curr_input, inputBuffer, sizeof(float) * nBufferFrames);
	pv->analysis();
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
  input_params.deviceId = 11;
  input_params.nChannels = channels;
  output_params.deviceId = 11;
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
 cout << "Offline Vocoding" << endl;

  AudioFile<float> * audioFile = new AudioFile<float>(); 
  AudioFile<float> outFile;
  if(!audioFile->load("/home/davis/Desktop/Phase-Vocoder/testtones/1000sine.wav"))
  {
    cout << "err: wav failed to load" << endl; 
    exit(1);
  }
  audioFile->printSummary();
  int numChannels = audioFile->getNumChannels();
  int numSamples = audioFile->getNumSamplesPerChannel();
  
  outFile.setAudioBufferSize(numChannels, phase->timeScale * numSamples);
  outFile.setSampleRate(44100);
  outFile.setBitDepth(16);
#ifdef TWOD
  float** d_input;
  cout << "Loading file...." << endl;
  cudaMallocManaged((void**)&d_input, sizeof(float*) * numChannels, cudaMemAttachHost);
  checkCUDAError_("Error creating unified memory for channels", __LINE__);
  for (int i = 0; i< numChannels; i++){
      cudaMallocManaged((void**)&d_input[i], audioFile->getNumSamplesPerChannel() * sizeof(float), cudaMemAttachHost);
      checkCUDAError_("Malloc Error: unified memory for samples", __LINE__);
      for (int j = 0; j < audioFile->getNumSamplesPerChannel(); j++){
          d_input[i][j] = audioFile->samples[i][j];
      }    
   }

  cout << "mallocing for output" << endl;
  float2*** d_output;
  cudaMallocManaged((void**)&d_output, sizeof(float2**) * numChannels, cudaMemAttachHost);
  checkCUDAError_("Error creating unified memory for channel output", __LINE__);
  for(int i = 0; i < numChannels; i++){
    cudaMallocManaged((void**)&d_output[i], sizeof(float2*) * numSamples / phase->hopSize, cudaMemAttachHost);
    checkCUDAError_("Erroring Mallocing 2D_analysis array", __LINE__);
    for(int j = 0; j < numSamples; j += phase->hopSize){
        //cudaMallocManaged((void**)&d_output[i][j], sizeof(float2) * phase->nSamps, cudaMemAttachHost); 
        cudaMalloc((void**)&d_output[i][j/phase->hopSize], sizeof(float2) * phase->nSamps); 
        checkCUDAError_("Erroring Mallocing analysis sample array", __LINE__);
    }
  }
  cout << "analysis..." << endl;
 //analysis
  float* intermediary;
  cudaMalloc((void**)&intermediary, sizeof(float) *phase->nSamps);
  for (int channel = 0; channel < numChannels; channel++){
	   cudaStreamAttachMemAsync(NULL, d_input[channel], 0, cudaMemAttachGlobal);
	   checkCUDAError_("attach input", __LINE__);
     for (int i = 0; i < numSamples - phase->hopSize; i += phase->hopSize) {
	       cudaStreamSynchronize(NULL);
	       phase->analysis(&d_input[channel][i / phase->hopSize],d_output[channel][i / phase->hopSize], intermediary);
         #ifdef DEBUG
          float2 *debug_arr;
          cudaMallocManaged((void**)&debug_arr, sizeof(float2) * phase->nSamps, cudaMemAttachHost);
          cudaMemcpy(debug_arr,d_output[channel][i/phase->hopSize], sizeof(float2) * phase->nSamps,cudaMemcpyDeviceToHost);
          printArraywNewLines(phase->nSamps, debug_arr);
          #endif
         checkCUDAError_("analysis error main.cpp", __LINE__);
      }
  }
  // create empty backFrame
  float* backFrame;
  cudaMallocManaged((void**)&backFrame, sizeof(float) *phase->nSamps, cudaMemAttachHost);
  checkCUDAError_("mallloc managed backframe main.cpp", __LINE__);
  for (int i = 0; i < phase->nSamps; i++) {
	  backFrame[i] = 0;
  }
  cout << "resynthesis" << endl;
  //resynthesis
  for (int channel = 0; channel < numChannels; channel++){
     for (int i = 0; i < numSamples - phase->hopSize; i += phase->hopSize) {
        float* final_output;
        cudaMallocManaged((void**)&final_output, sizeof(float) * phase->nSamps, cudaMemAttachHost);
        checkCUDAError_("mallloc managed main.cpp final_output", __LINE__);
	      phase->resynthesis(backFrame, d_output[channel][i/phase->hopSize], final_output);
	      cudaMemcpy(backFrame, final_output, sizeof(float) * phase->nSamps, cudaMemcpyHostToHost);
	      cudaFree(final_output);
        for(int j = 0; j < phase->outHopSize; j++){
            int idx = i /phase->hopSize * phase->outHopSize + j;
            outFile.samples[channel][idx] = backFrame[j];
        }
     }
  }
  cout << "writing to file" << endl;
  outFile.save("/home/davis/Desktop/Phase-Vocoder/output/out.wav");
  cudaFree(backFrame);
  for(int i = 0; i < numChannels; i++){
    for(int j = 0; j < numSamples / phase->hopSize; j++){
      cudaFree(d_output[i][j]);
    }
      cudaFree(d_input[i]);
      cudaFree(d_output[i]);
  }
#else 
  float* d_input;
  numChannels = 1;
  cudaMallocManaged((void**)&d_input, audioFile->getNumSamplesPerChannel() * sizeof(float), cudaMemAttachHost);
  checkCUDAError_("Malloc Error: unified memory for samples", __LINE__);
  for (int i = 0; i < audioFile->getNumSamplesPerChannel(); i++){
      d_input[i] = audioFile->samples[0][i];
  }    
  //save_results("/home/davis/Desktop/Phase-Vocoder/output.out", d_input, numSamples, 44100);
    checkCUDAError_("mem orr", __LINE__);
  //bufferToManageMemory(audioFile, d_input, 0);
  delete audioFile;
  cout << "mallocing for output" << endl;
  float2** d_output;
  cudaMallocManaged((void**)&d_output, sizeof(float2*) *  numSamples / phase->hopSize, cudaMemAttachHost);
  checkCUDAError_("Error creating unified memory for channel output", __LINE__);
  for(int j = 0; j < numSamples; j += phase->hopSize){
        cudaMalloc((void**)&d_output[j/ phase->hopSize], sizeof(float2) * phase->nSamps); 
        checkCUDAError_("Erroring Mallocing analysis sample array", __LINE__);
  }
  cout << "analysis..." << endl;
  //analysis
  cudaStreamAttachMemAsync(NULL, d_input, 0, cudaMemAttachGlobal);
  checkCUDAError_("malloc ini", __LINE__);
  for (int i = 0; i < numSamples - phase->hopSize; i += phase->hopSize) {
	       cudaStreamSynchronize(NULL);
	       phase->analysis(d_input,d_output[i / phase->hopSize], i);
         checkCUDAError_("analysis error main.cpp", __LINE__);
         #ifdef DEBUG
          float2 *debug_arr;
          cudaMallocManaged((void**)&debug_arr, sizeof(float2) * phase->nSamps, cudaMemAttachHost);
          cudaMemcpy(debug_arr,d_output[i/phase->hopSize], sizeof(float2) * phase->nSamps,cudaMemcpyDeviceToHost);
          printArraywNewLines(phase->nSamps, debug_arr);
         #endif
  }
  cout << "analysis over"  << endl;
  cudaFree(d_input);
  checkCUDAError_("freeing input", __LINE__);
  // create empty backFrame
  float* backFrame;
  cudaMallocManaged((void**)&backFrame, sizeof(float) * phase->nSamps, cudaMemAttachHost);
  checkCUDAError_("malloc managed backframe main.cpp", __LINE__);
  for (int i = 0; i < phase->nSamps; i++) {
	  backFrame[i] = 0;
  }
  cout << "resynthesis" << endl;
  //resynthesis
  for (int i = 0; i < numSamples - phase->hopSize; i += phase->hopSize) {
        float* final_output;
        cudaMallocManaged((void**)&final_output, sizeof(float) * phase->nSamps, cudaMemAttachHost);
        checkCUDAError_("mallloc managed main.cpp final_output", __LINE__);
	      phase->resynthesis(backFrame, d_output[i/phase->hopSize], final_output);
	      cudaMemcpy(backFrame, final_output, sizeof(float) * phase->nSamps, cudaMemcpyHostToHost);
        checkCUDAError_("Memcpy output to backframe", __LINE__);
	      cudaFree(final_output);
        checkCUDAError_("cudaFree final output", __LINE__);
        printArray(phase->nSamps, backFrame);
        for(int j = 0; j < phase->outHopSize; j++){
            int idx = i /phase->hopSize * phase->outHopSize + j;
            cout << idx << endl;
            outFile.samples[0][idx] = backFrame[j];
            outFile.samples[1][idx] = backFrame[j];
        }
  }
  cout << "writing to file" << endl;
  outFile.save("/home/davis/Desktop/Phase-Vocoder/output/out.wav");
  cudaFree(backFrame);
  for(int i = 0; i < numChannels; i++){
    for(int j = 0; j < numSamples / phase->hopSize; j++){
      cudaFree(d_output[j]);
    }
      cudaFree(d_input);
      cudaFree(d_output);
  } 
#endif
#endif //RT
 
  cudaFree(phase->imp);
  cufftDestroy(phase->plan);
  return 0;
}