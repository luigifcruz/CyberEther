#include "jetstream/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

Generic::Generic(Config& c) : Module(c.policy), cfg(c), in(c.input0) {
}

Result Generic::_initRender() {
    auto render = cfg.render;

    return Result::SUCCESS;
}

Result Generic::underlyingCompute() {
    return this->_compute();
}

Result Generic::underlyingPresent() {
    if (textureCfg.width != cfg.width || textureCfg.height != cfg.height) {
        if (surface->resize(cfg.width, cfg.height) != Render::Result::SUCCESS) {
            cfg.width = textureCfg.width;
            cfg.height = textureCfg.height;
        }
    }

    return this->_present();
}

} // namespace Jetstream::Waterfall

