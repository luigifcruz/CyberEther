#include "jetstream/waterfall/cpu.hpp"

namespace Jetstream::Waterfall {

CPU::CPU(const Config& c) : Generic(c) {
    ymax = cfg.size.height;
    bin.resize(in.buf.size() * ymax);
    JETSTREAM_CHECK_THROW(this->_initRender((uint8_t*)bin.data()));
}

CPU::~CPU() {
}

Result CPU::_compute() {
    std::copy(in.buf.begin(), in.buf.end(), bin.begin()+(inc * in.buf.size()));

    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
