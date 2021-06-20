#include "jetstream/histogram/generic.hpp"

namespace Jetstream::Histogram {

Generic::Generic(const Config& c) : Module(cfg.policy), cfg(c), in(cfg.input0) {}

std::weak_ptr<Render::Texture> Generic::tex() const {
    return texture;
}

} // namespace Jetstream::FFT

