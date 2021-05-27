#ifndef JETSTREAM_FFT_FFTW_H
#define JETSTREAM_FFT_FFTW_H

#include "jetstream/fft/generic.hpp"

#include <fftw3.h>

namespace Jetstream::FFT {

using I = cpu::arr::c32;
using O = cpu::arr::c32;

class CPU : public Generic {
public:
    explicit CPU(Config&, I&);
    ~CPU();

    constexpr O& out() {
        return output;
    };

protected:
    I& input;
    O output;

    Result underlyingCompute();
    Result underlyingPresent();

    // Complex Float
    fftwf_plan cf_plan;
};

} // namespace Jetstream::FFT

#endif
