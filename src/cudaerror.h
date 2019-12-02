#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
void checkCUDAError_(const char *msg, int line = -2) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= -1) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

