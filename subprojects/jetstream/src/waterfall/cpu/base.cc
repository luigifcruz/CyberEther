#include "jetstream/waterfall/cpu.hpp"

namespace Jetstream::Waterfall {

CPU::CPU(Config& c) : Generic(c) {
    auto render = cfg.render;

    ymax = cfg.height;
    bin.resize(in.buf.size() * ymax);

    binTextureCfg.buffer = (uint8_t*)bin.data();
    JETSTREAM_ASSERT_SUCCESS(this->_initRender());
}

CPU::~CPU() {
}

Result CPU::_compute() {
    std::copy(in.buf.begin(), in.buf.end(), bin.begin()+(inc * in.buf.size()));

    return Result::SUCCESS;
}

Result CPU::_present() {
    // TODO: hot garbage, fix
    int start = last;
    int blocks = (inc - last);

    if (blocks < 0) {
        blocks = ymax - last;
        binTexture->fill(start, 0, in.buf.size(), blocks);
        start = 0;
        blocks = inc;
    }

    binTexture->fill(start, 0, in.buf.size(), blocks);
    last = inc;

    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
