#ifndef JETSTREAM_FFT_BASE_H
#define JETSTREAM_FFT_BASE_H

#include "jetstream/fft/generic.hpp"
#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
#include "jetstream/fft/cpu.hpp"
#endif

namespace Jetstream::FFT {

#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
inline std::shared_ptr<CPU> Instantiate(Config& c, DF::CPU::CF32V& d) {
    return std::make_shared<CPU>(c, d);
}
#endif

} // namespace Jetstream::FFT

#endif
