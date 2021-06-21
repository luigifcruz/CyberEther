#ifndef JETSTREAM_FFT_BASE_H
#define JETSTREAM_FFT_BASE_H

#include "jetstream/fft/generic.hpp"
#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
#include "jetstream/fft/cpu.hpp"
#endif
#ifdef JETSTREAM_FFT_CUDA_AVAILABLE
#include "jetstream/fft/cuda.hpp"
#endif

#endif
