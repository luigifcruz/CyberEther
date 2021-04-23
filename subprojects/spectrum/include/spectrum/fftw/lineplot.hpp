#ifndef SPECTRUM_FFTW_LINEPLOT_H
#define SPECTRUM_FFTW_LINEPLOT_H

#include "spectrum/fftw/instance.hpp"

namespace Spectrum {

class FFTW::LinePlot : public Spectrum::LinePlot {
public:
    LinePlot(Config& cfg, FFTW& i) : Spectrum::LinePlot(cfg), inst(i) {};
    virtual ~LinePlot() = default;

    Result create();
    Result destroy();
    Result draw();

    uint raw();

protected:
    FFTW& inst;
};

} // namespace Render

#endif


