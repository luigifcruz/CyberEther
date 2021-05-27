#ifndef JETSTREAM_FFT_BASE_H
#define JETSTREAM_FFT_BASE_H

#include "jetstream/fft/generic.hpp"
#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
#include "jetstream/fft/cpu.hpp"
#endif

namespace Jetstream::FFT {

#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
inline std::shared_ptr<CPU> Instantiate(Config& config, I& input) {
    return std::make_shared<CPU>(config, input);
}
#endif

} // namespace Jetstream::FFT

#endif
