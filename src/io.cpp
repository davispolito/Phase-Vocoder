#include "io.h"
using namespace std;
void printArray(int n, float2 *a, bool abridged) {
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
void printArray(int n, float *a, bool abridged) {
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


void read_file(const char* filename, float* out, int count) {
  ifstream file;
  file.open(filename, fstream::binary | fstream::out);
  
  if (file.is_open()) {
	  int i = 0;
    while (i < count) {
      float val;
      if (file >> val) {
		  out[i++] = val;
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
  strcat(outfilename, buffer);
  strcat(outfilename, ".out");
  
  // Create the file
  ofstream outfile;
  outfile.open(outfilename, std::ofstream::out | std::ofstream::app);
  outfile.precision(4);
  
  // Save the data
  outfile << "frequency, value" << endl;
  for (int i = 0; i < count / 2; i++) {
      outfile << i * ((float)sample_rate/count) << "," << result[i].x << ", "<< result[i].y << endl;
  }
  
  outfile.close();
}

void save_results(const char* filename, float* result, size_t count, int sample_rate) {
  char* outfilename = new char[512];
  char* buffer = new char[10];
  // Compose the output filename
  strcpy(outfilename, filename);
  strcat(outfilename, ".out");
  
  // Create the file
  ofstream outfile;
  outfile.open(outfilename, std::ofstream::out);
  outfile.precision(4);
  
  // Save the data
  for (int i = 0; i < count; i++) {
      outfile << result[i]<< endl;
  }
  
  outfile.close();
}


void checkCUDAError_(const char *msg, int line) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= -1) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
