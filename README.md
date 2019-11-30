# Phase-Vocoder
  repo link 
  https://github.com/davispolito/Phase-Vocoder.git

## Branch System
#### cufft
  Tests of Cufft vs. GPU Cuda Algorithms developed from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.566.425&rep=rep1&type=pdf as well as one from a naive parallelization of Cooley-Tukey
   From the data obtained it was determined that Cufft was the fastest option available
![data](outputfft.png)
   Further discovery was made when porting the code to Jetson Nano that with GPU memory copies required to move CPU arrays onto the GPU that the Jetson Nano's unified memory was actually faster than the GTX 1060 that I originally tested on. Memory copy operations took over 0.4ms on the computer GPU and the entirety of Cufft runtime was 0.3ms on the Jetson Nano
#### nano-rt
   Here lies the code for realtime operation of the Phase Vocoding Algorithm on Jetson Nano.
   Current road block is this error
   `RtApiAlsa::getDeviceInfo: snd_pcm_open error for device (hw:1,0), Device or resource busy`
#### laptop-rt
   Forked from nano-rt this branch is being used to continue developing the real time algorithm for phase vocoding since nano-rt is facing issues with the linux/rasberry pi system. 
#### nano
   Used to compare the runtime of fft algorithms on the Jetson Nano vs. the Computer
  
## Project Motivation
   Phase Vocoding forms the basis of complex tools such as Autotune and the Prizmizer as well as Time stretching useful in quantization. These effects are commonly loaded offline, not allowing for real time manipulation or performance (Paul Stretch) or require computers to run the software ( Autotune, the Messina, etc.). By utilizing a GPU to parallelize the fft and frequency domain operations, this project aims to put the power of Phase Vocoding into the hands of all musicians. 
   
## The algorithm
  At the core of Phase Vocoding is the Fast Fourier Transform, a complex mathematical tool seperating time domain signals into their
discrete frequency components. Taking individual fourier transforms of samples that take into account parts of the frame before them we are
able to create a Short Time Fourier Transform Analysis that allows us to view and manipulate not only instantaneous phase but also phase
difference, representing the varying frequency components of a signal. After completing the analysis time stretching and pitch shifting is
able to be applied to the frequency content through operations as simple as vector scaling. These new vectors are then resynthesized using
an Inverse Fourier Tranform or Additive Synthesis. I hope to explore both options to determine which will work best in real time. 
