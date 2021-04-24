#include "spectrum/fftw/instance.hpp"
#include "spectrum/fftw/lineplot.hpp"

namespace Spectrum {

Result FFTW::create() {
    for (const auto& lineplot : cfg.lineplots) {
        lineplot->create();
    }

    return Result::SUCCESS;
}

Result FFTW::destroy() {
    for (const auto& lineplot : cfg.lineplots) {
        lineplot->destroy();
    }

    return Result::SUCCESS;
}

Result FFTW::feed() {
    for (const auto& lineplot : cfg.lineplots) {
        lineplot->draw();
    }

    return Result::SUCCESS;
}

std::shared_ptr<Spectrum::LinePlot> FFTW::create(Spectrum::LinePlot::Config& cfg) {
    auto lineplot = std::make_shared<FFTW::LinePlot>(cfg, *this);
    this->cfg.lineplots.push_back(lineplot);
    return lineplot;
}

} // namespace Spectrum
