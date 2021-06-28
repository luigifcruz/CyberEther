#ifndef JETSTREAM_FFT_BASE_H
#define JETSTREAM_FFT_BASE_H

#include "jetstream/modules/fft/generic.hpp"
#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
#include "jetstream/modules/fft/cpu.hpp"
#endif
#ifdef JETSTREAM_FFT_CUDA_AVAILABLE
#include "jetstream/modules/fft/cuda.hpp"
#endif

#endif
