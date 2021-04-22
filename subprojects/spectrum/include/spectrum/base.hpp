#ifndef SPECTRUM_H
#define SPECTRUM_H

#include "types.hpp"

#include "spectrum/base/instance.hpp"
#include "spectrum/base/lineplot.hpp"

#ifdef SPECTRUM_FFTW_AVAILABLE
#include "spectrum/fftw/instance.hpp"
#include "spectrum/fftw/lineplot.hpp"
#endif

namespace Spectrum {

inline std::vector<API> AvailableAPIs = {
#ifdef SPECTRUM_FFTW_AVAILABLE
    API::FFTW,
#endif
};

} // namespace Spectrum

#endif
