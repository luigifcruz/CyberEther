#ifndef JSTCORE_FFT_BASE_H
#define JSTCORE_FFT_BASE_H

#include "jstcore_config.hpp"
#include "jstcore/fft/generic.hpp"
#ifdef JSTCORE_FFT_FFTW_AVAILABLE
#include "jstcore/fft/cpu.hpp"
#endif
#ifdef JSTCORE_FFT_CUDA_AVAILABLE
#include "jstcore/fft/cuda/base.hpp"
#endif

#endif
