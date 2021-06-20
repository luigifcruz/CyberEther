#ifndef JETSTREAM_FFT_CUDA_H
#define JETSTREAM_FFT_CUDA_H

#include "jetstream/fft/generic.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream::FFT {

class CUDA : public Generic {
public:
    explicit CUDA(const Config &);
    ~CUDA();

protected:
    Result underlyingCompute() final;
    Result underlyingPresent() final;

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
