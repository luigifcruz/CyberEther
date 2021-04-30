#ifndef JETSTREAM_FFT_FFTW_H
#define JETSTREAM_FFT_FFTW_H

#include "jetstream/fft/generic.hpp"

#include <fftw3.h>

namespace Jetstream::FFT {

class CPU : public Generic {
public:
    explicit CPU(Config& c);
    ~CPU();

protected:
    Result underlyingCompute();
    Result underlyingPresent();

    // Complex Float
    fftwf_plan cf_plan;
};

} // namespace Jetstream::FFT

#endif

