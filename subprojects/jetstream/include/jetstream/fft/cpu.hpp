#ifndef JETSTREAM_FFT_FFTW_H
#define JETSTREAM_FFT_FFTW_H

#include "jetstream/fft/config.hpp"

#include <fftw3.h>

namespace Jetstream::FFT {

class CPU : public Transform {
public:
    explicit CPU(Config& c);
    ~CPU();

protected:
    Config& cfg;

    Result underlyingCompute();
    Result underlyingPresent();

    // Complex Float
    fftwf_plan cf_plan;
};

} // namespace Jetstream::FFT

#endif

