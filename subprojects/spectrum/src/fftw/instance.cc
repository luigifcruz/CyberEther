#include "spectrum/fftw/instance.hpp"
#include "spectrum/fftw/lineplot.hpp"

namespace Spectrum {

Result FFTW::create() {
    fft_out = (std::complex<float>*)malloc(sizeof(std::complex<float>) * cfg.size);
    fft_plan = fftwf_plan_dft_1d(cfg.size, reinterpret_cast<fftwf_complex*>(cfg.buffer),
            reinterpret_cast<fftwf_complex*>(fft_out), FFTW_FORWARD, FFTW_MEASURE);

    for (const auto& lineplot : cfg.lineplots) {
        lineplot->create();
    }

    return Result::SUCCESS;
}

Result FFTW::destroy() {
    fftwf_destroy_plan(fft_plan);

    for (const auto& lineplot : cfg.lineplots) {
        lineplot->destroy();
    }

    return Result::SUCCESS;
}

Result FFTW::feed() {
    fftwf_execute(fft_plan);

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
