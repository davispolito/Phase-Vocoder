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
#include "testing_helpers.hpp"
#include <iomanip>
#include "Sine.h"
#include "phaseVocoder.h"
using namespace std;

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
  float input[2048];
  for (int i = 0; i < 2048; i++){
    input[i] = sine->tick();
  }
  PhaseVocoder* phase = new PhaseVocoder();
  float2* output;
  cudaMallocManaged((void**)&output, sizeof(float2*) * phase->N/2 * phase->N / phase->R);
  phase->analysis(input, output);
	return 0;
}
