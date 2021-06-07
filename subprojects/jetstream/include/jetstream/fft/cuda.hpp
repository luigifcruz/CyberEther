#ifndef JETSTREAM_FFT_CUDA_H
#define JETSTREAM_FFT_CUDA_H

#include "jetstream/fft/generic.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream::FFT {

class CUDA : public Generic {
public:
    explicit CUDA(Config&);
    ~CUDA();

protected:
    Result underlyingCompute();
    Result underlyingPresent();

    cufftHandle plan;
    cufftComplex* fft_dptr;

    nonstd::span<std::complex<float>> fft;
    std::vector<float> buf_out;
};

} // namespace Jetstream::FFT

#endif
