#ifndef SPECTRUM_FFTW_INSTANCE_H
#define SPECTRUM_FFTW_INSTANCE_H

#include "spectrum/base/instance.hpp"

namespace Spectrum {

class FFTW : public Spectrum::Instance {
public:
    class LinePlot;

    FFTW(Config& c) : Spectrum::Instance(c) {};

    Result create();
    Result destroy();

    Result feed();

protected:
};

struct State : Spectrum::FFTW {
    std::shared_ptr<Render::Instance> render;
};

} // namespace Spectrum

#endif
