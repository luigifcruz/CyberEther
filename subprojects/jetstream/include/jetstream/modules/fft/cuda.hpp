#ifndef JETSTREAM_FFT_CUDA_H
#define JETSTREAM_FFT_CUDA_H

#include "jetstream/modules/fft/generic.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream {

class FFT::CUDA : public FFT {
public:
    explicit CUDA(const Config & cfg, IO & input);
    ~CUDA();

protected:
    Result compute() final;
    Result present() final;

    cufftHandle plan;
    cudaStream_t stream;

    size_t fft_len;
    cufftComplex* fft_dptr;

    size_t out_len;
    float* out_dptr;

    size_t win_len;
    cufftComplex* win_dptr;
};

} // namespace Jetstream

#endif
