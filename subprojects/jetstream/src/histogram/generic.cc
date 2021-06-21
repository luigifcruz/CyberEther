#include "jetstream/histogram/generic.hpp"

namespace Jetstream {

Histogram::Histogram(const Config& c) : Module(cfg.policy), cfg(c), in(cfg.input0) {}

std::weak_ptr<Render::Texture> Histogram::tex() const {
    return texture;
}

} // namespace Jetstream
