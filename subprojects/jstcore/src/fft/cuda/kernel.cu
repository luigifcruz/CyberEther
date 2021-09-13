#include "jstcore/fft/cuda/kernel.hpp"

namespace Jetstream {
namespace FFT {
namespace Kernel {

static __device__ inline float clamp(const float x, const float a, float b) {
    return (x < a) ? a : (b < x) ? b : x;
}

static __device__ inline float scale(const float x, const float min, const float max) {
    return (x - min) / (max - min);
}

static __device__ inline float amplt(const cuFloatComplex x, const int n) {
    return 20 * log10(cuCabsf(x) / n);
}

static __global__ void pre(cufftComplex* c, const cufftComplex* win, const uint n){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < n; i += numThreads) {
        c[i] = cuCmulf(c[i], win[i]);
    }
}

static __global__ void post(const cufftComplex* c, float* r,
    const float min, const float max, const uint n){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp;
    for (int i = threadID; i < n; i += numThreads) {
        tmp = amplt(c[i], n);
        tmp = scale(tmp, min, max);
        tmp = clamp(tmp, 0.0f, 1.0f);

        r[i] = tmp;
    }
}

void PreFFT(const int grid, const int block, const cudaStream_t cuda_stream,
        cufftComplex* fft, const cufftComplex* win_in, const int len) {
    pre<<<grid, block, 0, cuda_stream>>>(fft, win_in, len);
}

void PostFFT(const int grid, const int block, const cudaStream_t cuda_stream,
        const cufftComplex* fft_in, float* fft_out, const float min, const float max, const int len) {
    post<<<grid, block, 0, cuda_stream>>>(fft_in, fft_out, min, max, len);
}

} // namespace Kernel
} // namespace FFT
} // namespace Jetstream
