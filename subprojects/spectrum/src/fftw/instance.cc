#include "spectrum/fftw/instance.hpp"
#include "spectrum/fftw/lineplot.hpp"

namespace Spectrum {

Result FFTW::create() {
    state = (State*)malloc(sizeof(State));
    std::cout << (state == nullptr) << std::endl;
    state->render = cfg.render;

    for (const auto& lineplot : cfg.lineplots) {
        lineplot->create();
    }

    return Result::SUCCESS;
}

Result FFTW::destroy() {
    for (const auto& lineplot : cfg.lineplots) {
        lineplot->destroy();
    }

    free(state);

    return Result::SUCCESS;
}

Result FFTW::feed() {
    return Result::SUCCESS;
}

} // namespace Spectrum
