#include "jstcore/waterfall/cpu.hpp"

namespace Jetstream::Waterfall {

CPU::CPU(const Config & config, const Input & input) : Generic(config, input) {
    ymax = config.size.height;
    bin.resize(input.in.buf.size() * ymax);
    this->initRender((uint8_t*)bin.data());
}

Result CPU::underlyingCompute() {
    std::copy(input.in.buf.begin(), input.in.buf.end(), bin.begin()+(inc * input.in.buf.size()));
    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
