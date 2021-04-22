#ifndef SPECTRUM_FFTW_LINEPLOT_H
#define SPECTRUM_FFTW_LINEPLOT_H

#include "spectrum/fftw/instance.hpp"

namespace Spectrum {

class FFTW::LinePlot : public Spectrum::LinePlot {
public:
    LinePlot(Config& cfg, State& s) : Spectrum::LinePlot(cfg), state(s) {};
    virtual ~LinePlot() = default;

    Result create();
    Result destroy();
    Result draw();

    uint raw();

protected:
    State& state;
    std::shared_ptr<Render::Texture> texture;
};

} // namespace Render

#endif


