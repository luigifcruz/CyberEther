#ifndef JSTCORE_FFT_CPU_H
#define JSTCORE_FFT_CPU_H

#include "jstcore/fft/generic.hpp"

#include <fftw3.h>

namespace Jetstream::FFT {

template<>
class Backend<Device::CPU> : public Generic  {
public:
    explicit Backend(const Config&, const Input&);
    ~Backend();

protected:
    const Result underlyingCompute();

    fftwf_plan cf_plan;

    std::vector<std::complex<float>> fft_in;
    std::vector<std::complex<float>> fft_out;
    std::vector<float> amp_out;
};

} // namespace Jetstream::FFT

#endif
