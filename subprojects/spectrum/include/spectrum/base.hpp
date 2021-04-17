#ifndef SPECTRUM_H
#define SPECTRUM_H

#include "types.hpp"

#ifdef SPECTRUM_FFTW_AVAILABLE
#endif

namespace Spectrum {

inline std::vector<API> AvailableAPIs = {
#ifdef SPECTRUM_FFTW_AVAILABLE
    API::FFTW,
#endif
};

} // namespace Spectrum

#endif
