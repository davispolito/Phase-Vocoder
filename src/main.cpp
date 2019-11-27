#include "RtAudio.h"
#include "kernel.h"
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstring>
#include "cufft_.h"
#include <iomanip>
#include "phaseVocoder.h"
#include "RtError.h"


using namespace std;
void printArray(int n, float2 *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("{%3f, ", a[i].x);
        printf("%3f},", a[i].y);
    }
    printf("]\n");
}
void printArray(int n, float *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3f, ", a[i]);
    }
    printf("]\n");
}

void read_file(const char* filename, vector<float2>& out) {
  ifstream file;
  file.open(filename, fstream::binary | fstream::out);
  
  if (file.is_open()) {
    while (!file.eof()) {
      float2 val;
      if (file >> val.x) {
        val.y = 0;
        out.push_back(val);
      }
    }
  } else {
    cerr << "Can't open file " << filename << " for reading." << endl;
  }
  
  file.close();
}


void read_file(const char* filename, vector<float>& out) {
  ifstream file;
  file.open(filename, fstream::binary | fstream::out);
  
  if (file.is_open()) {
    while (!file.eof()) {
      float val;
      if (file >> val) {
        out.push_back(val);
      }
    }
  } else {
    cerr << "Can't open file " << filename << " for reading." << endl;
  }
  
  file.close();
}

/**
 * Saves the result data to an output file.
 */
void save_results(const char* filename, float2* result, size_t count, int sample_rate) {
  char* outfilename = new char[512];
  char* buffer = new char[10];
  // Compose the output filename
  strcpy(outfilename, filename);
  sprintf(buffer, "%d", count);
  strcat(outfilename, buffer);
  strcat(outfilename, ".out");
  
  // Create the file
  ofstream outfile;
  outfile.open (outfilename);
  outfile.precision(4);
  
  // Save the data
  outfile << "frequency, value" << endl;
  for (int i = 0; i < count / 2; i++) {
      outfile << i * ((float)sample_rate/count) << "," << result[i].x << ", "<< result[i].y << endl;
  }
  
  outfile.close();
}

void compute_file(const char *filename,vector<float2> buffer, size_t threads, int radix, int num_samples) {
	cout << filename << ", " << "threads = " << threads << ", radix = " << radix
		<< ", num_samples" << num_samples << endl; 
}

int callback(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames, double streamTime, 
            RtAudioStreamStatus status, void *data) {
  if (status){
    std::cout << "Stream overflow deteced" << std::endl;
  }

  unsigned long *bytes = (unsigned long *) data;
  memcpy(outputBuffer, inputBuffer, *bytes);
  return 0;
}

int main()
{
  float* input;
  cudaMallocManaged((void**)&input, sizeof(float) * 2048, cudaMemAttachHost);

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
  input_params.deviceId = 11;
  input_params.nChannels = 2;
  output_params.deviceId = 11;
  output_params.nChannels = 2;
  PhaseVocoder* phase = new PhaseVocoder(256);
  float2* output_2D[2048 / 64];
  float2* magFreq_2D[2048/64];
  float2* output, *magFreq;
  cudaMallocManaged((void**)&output, sizeof(float2) * 2 * phase->nSamps);
  cudaMallocManaged((void**)&magFreq, sizeof(float2) * 2 * phase->nSamps);
  //phase->analysis(&input[0], output, magFreq);
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
  cudaFree(output);
  cudaFree(magFreq);
  cudaFree(phase->imp);
  cufftDestroy(phase->plan);
	return 0;
}
