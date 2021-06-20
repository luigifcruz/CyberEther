#ifndef JETSTREAM_FFT_BASE_H
#define JETSTREAM_FFT_BASE_H

#include "jetstream/fft/generic.hpp"
#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
#include "jetstream/fft/cpu.hpp"
#endif
#ifdef JETSTREAM_FFT_CUDA_AVAILABLE
#include "jetstream/fft/cuda.hpp"
#endif

namespace Jetstream::FFT {

inline std::shared_ptr<Generic> Instantiate(Locale L, const Config & config) {
    switch (L) {
#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
        case Jetstream::Locale::CPU:
            return std::make_shared<CPU>(config);
#endif
#ifdef JETSTREAM_FFT_CUDA_AVAILABLE
        case Jetstream::Locale::CUDA:
            return std::make_shared<CUDA>(config);
#endif
        default:
            JETSTREAM_CHECK_THROW(Jetstream::Result::ERROR);
    }
}

} // namespace Jetstream::FFT

#endif
