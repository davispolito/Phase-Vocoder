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
#include "Sine.h"
#include "phaseVocoder.h"

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
      outfile << i * ((float)sample_rate/count) << "," << result[i].x << endl;
  }
  
  outfile.close();
}

void compute_file(const char *filename,vector<float2> buffer, size_t threads, int radix, int num_samples) {
	cout << filename << ", " << "threads = " << threads << ", radix = " << radix
		<< ", num_samples" << num_samples << endl; 
}

int main()
{
  Sine* sine = new Sine();
  sine->setSamplingRate(44100);
  sine->setFrequency(10000);
  float* input;
  cudaMallocManaged((void**)&input, sizeof(float) * 2048, cudaMemAttachHost);
  for (int i = 0; i < 2048; i++){
    input[i] = sine->tick();
  }
  PhaseVocoder* phase = new PhaseVocoder(256);
  float2* output_2D[2048 / 64];
  //output_2D = malloc(sizeof(float2*) * 2048 / 64);
  float2* magFreq_2D[2048/64];
  //magFreq_2D = malloc(sizeof(float2*) * 2048 / 64); 
  int i;
  //for(i = 0; (i + 256) < 2048; i = i + 64 ){
      float2* output, *magFreq;
      cudaMallocManaged((void**)&output, sizeof(float2) * 2 * phase->nSamps);
      cudaMallocManaged((void**)&magFreq, sizeof(float2) * 2 * phase->nSamps);
      phase->analysis(&input[0], output, magFreq);
      printf("here%d\n",i);
      //printArray(2 * phase->nSamps, output);
      //printArray(2 * phase->nSamps, magFreq);
      output_2D[i/64] = output;
      magFreq_2D[i/64] = magFreq;
  //}
for(int k = 0; k < 1; k++){
  save_results("/home/davis/Desktop/Phase-Vocoder/data/magPhase", magFreq_2D[k], k, 44100);
  save_results("/home/davis/Desktop/Phase-Vocoder/data/output", output_2D[k], k, 44100);
  cudaFree(output_2D[k]);
  cudaFree(magFreq_2D[k]);
}
  
cudaFree(phase->imp);

	return 0;
}
