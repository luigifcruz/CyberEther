#ifndef JETSTREAM_FFT_FFTW_H
#define JETSTREAM_FFT_FFTW_H

#include "jetstream/fft/generic.hpp"

#include <fftw3.h>

namespace Jetstream::FFT {

class CPU : public Generic {
public:
    explicit CPU(Config&);
    ~CPU();

protected:
    Result underlyingCompute() final;
    Result underlyingPresent() final;

    fftwf_plan cf_plan;

    std::vector<std::complex<float>> fft_in;
    std::vector<std::complex<float>> fft_out;
    std::vector<float> amp_out;
};

} // namespace Jetstream::FFT

#endif
