#ifndef JETSTREAM_FFT_BASE_H
#define JETSTREAM_FFT_BASE_H

#include "jetstream/fft/generic.hpp"
#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
#include "jetstream/fft/cpu.hpp"
#endif

namespace Jetstream::FFT {

inline std::shared_ptr<Generic> Instantiate(Locale L, Config& config) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<CPU>(config);
        default:
            JETSTREAM_ASSERT_SUCCESS(Result::ERROR);
    }
}

} // namespace Jetstream::FFT

#endif
