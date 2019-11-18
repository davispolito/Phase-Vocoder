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
#include "hpfft.h"

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
  
  // Compose the output filename
  strcpy(outfilename, filename);
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
	printf("\n");
    printf("****************\n");
    printf("** FFT TESTS **\n");
    printf("****************\n");
	
	vector<float2> HZ_50;
	vector<float2> HZ_50_500;
	vector<float2> HZ_50_505_12000;
	
	read_file("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/a.txt", HZ_50);
	read_file("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/50Hz+500Hz/512smp@44100.dat", HZ_50_500);
	//read_file("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/50Hz+505Hz+12000Hz/512smp@44100.dat", HZ_50_505_12000);

	int size_50 = HZ_50.size();
	int size_50_500 = HZ_50_500.size();
	int size_50_505_12000 = HZ_50_505_12000.size();

	int r, threads;
	float computations = 0;
	float sum = 0;
	/*for(threads = 1; threads <= 1024; threads <<= 1){
		for (int n = 32; n <= 1024; n <<= 1) {
			cout << "50 Hz Wave, " << n << "samples" << endl;
	        float2* a = FFT::CuFFT::computeCuFFT(&HZ_50[0], n);
			printElapsedTime(FFT::CuFFT::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
			cout << "50, 500 Hz Wave, " << n << "samples" << endl;
	        float2* b = FFT::CuFFT::computeCuFFT(&HZ_50_500[0], n);
			printElapsedTime(FFT::CuFFT::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
			cout << "50, 505, 12000 Hz Wave, " << n << "samples" << endl;
	//        FFT::CuFFT::computeCuFFT(&HZ_50_505_12000[0], n);
		//	printElapsedTime(FFT::CuFFT::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		}
	}*/

	float2* a = FFT::CuFFT::computeCuFFT(&HZ_50[0], 512);
	float2* b = FFT::HPFFT::computeFFTSh(&HZ_50[0], 512, 2, 256);
	float2* c = FFT::HPFFT::computeFFTCooley(&HZ_50[0], 512, 2, 512);
	save_results("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/cufftout", a, 512, 44100);
	save_results("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/hpfftout", b, 512, 44100);
	save_results("C:/Users/Davis/Desktop/Vocoder/Phase-Vocoder/src/hpfftoutc", c, 512, 44100);
	

	for (threads = 1; threads <= 1024; threads <<= 1) {
		for (int n = 32; n <= 1024; n <<= 1) {

		}
	}
	return 0;
}
