#ifndef JSTCORE_FFT_CUDA_KERNEL_H
#define JSTCORE_FFT_CUDA_KERNEL_H

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream {
namespace FFT {
namespace Kernel {

void PreFFT(const int grid, const int block, const cudaStream_t cuda_stream,
        cufftComplex* fft, const cufftComplex* win_in, const int len);
void PostFFT(const int grid, const int block, const cudaStream_t cuda_stream,
        const cufftComplex* fft_in, float* fft_out, const float min, const float max, const int len);

} // namespace Kernel
} // namespace FFT
} // namespace Jetstream

#endif
