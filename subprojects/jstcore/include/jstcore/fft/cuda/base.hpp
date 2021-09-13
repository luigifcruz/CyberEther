#ifndef JSTCORE_FFT_CUDA_BASE_H
#define JSTCORE_FFT_CUDA_BASE_H

#include "jstcore/fft/generic.hpp"
#include "jstcore/fft/cuda/kernel.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream::FFT {

class CUDA : public Generic  {
public:
    explicit CUDA(const Config &, const Input &);
    ~CUDA();

protected:
    Result underlyingCompute() final;

    cufftHandle plan;
    cudaStream_t stream;

    size_t fft_len;
    cufftComplex* fft_dptr;

    size_t out_len;
    float* out_dptr;

    size_t win_len;
    cufftComplex* win_dptr;
};

} // namespace Jetstream::FFT

#endif

