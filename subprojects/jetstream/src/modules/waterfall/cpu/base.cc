#include "jetstream/modules/waterfall/cpu.hpp"

namespace Jetstream {

Waterfall::CPU::CPU(const Config& cfg, IO & input) : Waterfall(cfg, input) {
    ymax = cfg.size.height;
    bin.resize(in.buf.size() * ymax);
    JETSTREAM_CHECK_THROW(this->_initRender((uint8_t*)bin.data()));
}

Waterfall::CPU::~CPU() {
}

Result Waterfall::CPU::_compute() {
    std::copy(in.buf.begin(), in.buf.end(), bin.begin()+(inc * in.buf.size()));

    return Result::SUCCESS;
}

} // namespace Jetstream
