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
    Result underlyingCompute();
    Result underlyingPresent();

    // Complex Float
    fftwf_plan cf_plan;
    std::vector<std::complex<float>> buf_fft;
    std::vector<float> buf_out;

    static inline float log10f_fast(float);
};

} // namespace Jetstream::FFT

#endif
