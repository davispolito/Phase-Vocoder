#pragma once
#include "RtAudio.h"
#include "kernel.h"
#include "cufft_.h"
#include "phaseVocoder.h"
#include "RtError.h"
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

void printArray(int n, float2 *a, bool abridged = false); 
void printArraywNewLines(int n, float2 *a, bool abridged = false); 
void printArraywNewLines(int n, float *a, bool abridged = false); 
void printArray(int n, float *a, bool abridged = false); 
void read_file(const char* filename, float* out, int count); 
void save_results(const char* filename, float2* result, size_t count, int sample_rate);
void save_results(const char* filename, float* result, size_t count, int sample_rate); 
void checkCUDAError_(const char *msg, int line = -1);
