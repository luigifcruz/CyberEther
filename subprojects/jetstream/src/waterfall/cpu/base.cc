#include "jetstream/waterfall/cpu.hpp"

namespace Jetstream::Waterfall {

CPU::CPU(Config& c) : Generic(c) {
    auto render = cfg.render;

    ymax = cfg.height;
    bin.resize(in.buf.size() * ymax);

    binTextureCfg.buffer = (uint8_t*)bin.data();
    JETSTREAM_CHECK_THROW(this->_initRender());
}

CPU::~CPU() {
}

Result CPU::_compute() {
    std::copy(in.buf.begin(), in.buf.end(), bin.begin()+(inc * in.buf.size()));

    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
