#ifndef SPECTRUM_FFTW_API_H
#define SPECTRUM_FFTW_API_H

#include "spectrum/base/instance.hpp"

namespace Spectrum {

class FFTW {
public:
    class Instance;

    FFTW();

    std::shared_ptr<Instance> createInstance(Spectrum::Instance::Config&);

private:
    struct State {
    };

    State state;
    std::shared_ptr<Instance> instance;
};

} // namespace Spectrum

#endif
