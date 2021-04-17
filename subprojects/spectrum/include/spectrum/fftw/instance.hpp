#ifndef SPECTRUM_FFTW_INSTANCE_H
#define SPECTRUM_FFTW_INSTANCE_H

#include "spectrum/base/instance.hpp"
#include "spectrum/fftw/api.hpp"

namespace Spectrum {

class FFTW::Instance : public Spectrum::Instance {
public:
    Instance(Config& c, State& s) : Spectrum::Instance(c), state(s) {};

private:
    State& state;
};

} // namespace Spectrum

#endif
