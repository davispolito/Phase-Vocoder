#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include "hpfft.h"
#define TILE_SIZE 512
# define M_PI           3.14159265358979323846  /* pi */
namespace FFT {
	namespace HPFFT {
		using FFT::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__device__ inline float2 ComplexMul(float2 a, float2 b) {

			float2 c;

			c.x = a.x * b.x - a.y * b.y;

			c.y = a.x * b.y + a.y * b.x;

			return c;

		}

		__device__
			float2 eIThetta(int k, int N, int n, int offset) {
			float re = cos((2 * M_PI * (2 * n + offset) * k) / N);
			float im = (-1) * sin((2 * M_PI * (2 * n + offset) * k) / N);
			return { re, im };
		}

		__global__ void FFTCooley(float2* d_signal, float2* out, int N) {
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx > N) {
				return;
			}
			__shared__ float2 numbers_SM[TILE_SIZE];

			float2 even;
			even.x = 0;
			even.y = 0;
			float2 odd;
			odd.x = 0;
			odd.y = 0;
			numbers_SM[threadIdx.x] = d_signal[threadIdx.x];
			for (int n = 0; n < (TILE_SIZE / 2); n++) {
				float2 comp = numbers_SM[2 * n];
				float2 eThetta = eIThetta(idx, N, (n + ((TILE_SIZE / 2))), 0);
				float2 resultEven = ComplexMul(comp, eThetta);
				even.x = resultEven.x + even.x;
				even.y = resultEven.y + even.y;

				// compute the odd part
				float2 compOdd = numbers_SM[2 * n + 1];

				float2 eThettaOdd = eIThetta(idx, N, (n + ((TILE_SIZE / 2))), 1);

				float2 resultOdd = ComplexMul(compOdd, eThettaOdd);
				odd.x = resultOdd.x + odd.x;
				odd.y = resultOdd.y + odd.y;
			}
			// make sure all threads computed current phase
			__syncthreads();
			out[idx] = { odd.x + even.x, odd.y + even.y };
		}

		__device__	int expand(int idxL, int N1, int N2) {
			return (idxL / N1)*N1*N2 + (idxL%N1);
		}
		__device__ void exchange(float2* v, int R, int stride,
			int idxD, int incD,
			int idxS, int incS) {
			__shared__ float sr[4096];
			float *si = sr + incS * R;
			__syncthreads();

			for (int r = 0; r < R; r++) {
				int i = (idxD + r * incD) * stride;
				sr[i] = v[r].x;
				si[i] = v[r].y;
			}
			__syncthreads();

			for (int r = 0; r < R; r++) {
				int i = (idxS + r * incS) *stride;
				v[r] = { sr[i], si[i] };
			}

		}


		__device__ void FFT2(float2* v) {
			float2 v0 = v[0];
			v[0].x = v0.x + v[1].x;
			v[0].y = v0.y + v[1].y;
			v[1].x = v0.x - v[1].x;
			v[1].y = v0.y - v[1].y;
		}


		__device__ void DoFft(float2* v, int R, int N, int j, int sign, int numThreads, int stride = 1) {
			for (int Ns = 1; Ns < N; Ns *= R) {
				float angle = sign * 2 * M_PI * (float)(j % Ns) / (float)(Ns *R);
				for (int r = 0; r < R; r++) {
					v[r] = ComplexMul(v[r], { cos(r*angle), sin(r*angle) });
				}
				FFT2(v);
				int idxD = expand(j, Ns, R);
				int idxS = expand(j, N / R, R);
				exchange(v, R, stride, idxD, Ns, idxS, N / R);
			}
		}
		/**
		*
		*/
		__global__ void FFTShMem(float2* d_signal, int N, int numThreads, int sign, int R = 2) {
			int idx = threadIdx.x + (blockDim.x * blockIdx.x);
			if (idx >= N)
			{
				return;
			}
			float2 v[2];

			for (int r = 0; r < R; r++) {
				v[r] = d_signal[idx + r * numThreads];
			}
			if (numThreads == N / R) {
				DoFft(v, R, N, idx, sign, numThreads);
			}
			else {
				int idx2 = expand(threadIdx.x, N / R, R);
				exchange(v, R, 1, idx2, N / R, threadIdx.x, numThreads);
				DoFft(v, R, N, threadIdx.x, sign, numThreads);
				exchange(v, R, 1, threadIdx.x, numThreads, idx2, N / R);
			}
			float s = (sign < 1) ? 1 : 1 / N;
			for (int r = 0; r < R; r++) {
				d_signal[idx + r * numThreads] = { s * v[r].x, s*v[r].y };
			}
		}

		__device__ void FftIteration(int j, int N, int R, int Ns, float2* d_signal, float2* o_signal, int sign) {
			float2 v[2];
			int idxS = j; 
			float angle = -sign * 2 * M_PI*(j%Ns) / (Ns*R);
			for (int r = 0; r < R; r++) {
				v[r] = d_signal[idxS + r * N / R];
				v[r] = ComplexMul(v[r], { cos(r*angle), sin(r*angle) });
			}

			FFT2(v);
			int idxD = expand(j, Ns, R);
			for (int r = 0; r < R; r++) {
				o_signal[idxD + r * Ns] = v[r];
			}
		}

		__global__ void GPU_FFT(int N, int R, int Ns, float2* d_signal, float2* o_signal, int sign = 1) {
			int idx = blockDim.x * blockIdx.x + threadIdx.x;
			if (idx >= N) {
				return;
			}
			FftIteration(idx, N, R, Ns, d_signal, o_signal, sign);
		}

		void swap(float2* &a, float2* &b){
			 float2 *temp = a;
			 a = b;
			 b = temp;
		}

		void computeGPUFFT_RT(int N, int R, float2* h_signal, float2* intermediary, cudaStream_t* stream) {
			int blocksPerGrid = 1;
			for (int Ns = 1; Ns < N; Ns *= R) {
				GPU_FFT << <blocksPerGrid, N / R, 0 , *stream >> > (N, R, Ns, h_signal, intermediary);
				 swap(h_signal, intermediary);
			}
		}
		void computeGPUIFFT_RT(int N, int R, float2* h_signal, float2* intermediary, cudaStream_t* stream) {
			int blocksPerGrid = 1;
			for (int Ns = 1; Ns < N; Ns *= R) {
				GPU_FFT << <blocksPerGrid, N / R, 0 , *stream >> > (N, R, Ns, h_signal, intermediary, -1);
				 swap(h_signal, intermediary);
			}
		}

		void computeGPUFFT(int N, int R, float2* h_signal, float2* intermediary) {
			int blocksPerGrid = 1;
			for (int Ns = 1; Ns < N; Ns *= R) {
				GPU_FFT << <blocksPerGrid, N / R >> > (N, R, Ns, h_signal, intermediary);
				 swap(h_signal, intermediary);
			}
		}
		void computeGPUIFFT(int N, int R, float2* h_signal, float2* intermediary) {
			int blocksPerGrid = 1;
			for (int Ns = 1; Ns < N; Ns *= R) {
				GPU_FFT << <blocksPerGrid, N / R >> > (N, R, Ns, h_signal, intermediary, -1);
				 swap(h_signal, intermediary);
			}
		}

		float2* computeFFTSh(float2* h_signal, int N, int R, int numThreads) {
			float2* d_signal;
			//int blocksPerGrid = (N + numThreads - 1) / numThreads;
			int blocksPerGrid = 1;
			cudaMalloc((void**)&d_signal, sizeof(float2*)*N);
			cudaMemcpy(d_signal, h_signal, N * sizeof(float2), cudaMemcpyHostToDevice);
			FFTShMem << <blocksPerGrid, numThreads >> > (d_signal, N, numThreads, 1);

			float2 *o_signal;
			o_signal = (float2*)malloc(N * sizeof(float2));
			cudaMemcpy(o_signal, d_signal, sizeof(float2) * N, cudaMemcpyDeviceToHost);
			cudaFree(d_signal);
			return o_signal;
		}

		float2* computeFFTCooley(float2* h_signal, int N, int R, int numThreads) {
			float2* d_signal, *out;
			//int blocksPerGrid = (N + numThreads - 1) / numThreads;
			int blocksPerGrid = 1;
			cudaMalloc((void**)&d_signal, sizeof(float2*)*N);
			cudaMalloc((void**)&out, sizeof(float2*)*N);
			cudaMemcpy(d_signal, h_signal, N * sizeof(float2), cudaMemcpyHostToDevice);
			FFTCooley << <blocksPerGrid, numThreads >> > (d_signal, out, N);

			float2 *o_signal;
			o_signal = (float2*)malloc(N * sizeof(float2));
			cudaMemcpy(o_signal, out, sizeof(float2) * N, cudaMemcpyDeviceToHost);
			cudaFree(d_signal);
			cudaFree(out);
			return o_signal;
		}
	}
}
