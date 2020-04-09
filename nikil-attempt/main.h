// Include C++ libraries
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

// Include special libaries
#include "RtAudio.h"
#include "RtError.h"
#include "AudioFile.h"

// Include code we wrote

// Sets up the program
int main(int argc, char **argv);

// This is the live loop
// that acquires samples,
// listens for input (a target time or frequency shift)
// analyzes (STFT),
// time or frequency shifts,
// resynthesizes (iSTFT) back into time domain
void eventLoop();

// Short-Time Fourier Transform. Something like this for input
// This will call the cuda kernel function
// might need to change this to a normal array
std::vector<uint16_t> analyze(std::vector<uint16_t> time_domain);

// inverse Short-Time Fourier Transform
// might need to change this to a normal array
std::vector<uint16_t> synthesize(std::vector<uint16_t> frequency_domain);

