#ifndef JETSTREAM_FFT_FFTW_H
#define JETSTREAM_FFT_FFTW_H

#include "jetstream/fft/generic.hpp"

#include <fftw3.h>

namespace Jetstream {

class FFT::CPU : public FFT {
public:
    explicit CPU(const Config &, Manifest &);
    ~CPU();

protected:
    Result compute() final;
    Result present() final;

    fftwf_plan cf_plan;

    std::vector<std::complex<float>> fft_in;
    std::vector<std::complex<float>> fft_out;
    std::vector<float> amp_out;
};

} // namespace Jetstream

#endif
