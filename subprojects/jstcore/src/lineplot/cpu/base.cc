#include "jstcore/lineplot/cpu.hpp"

namespace Jetstream::Lineplot {

CPU::CPU(const Config & config, const Input & input) : Generic(config, input) {
    this->initRender(plot.data());
}

Result CPU::underlyingCompute() {
    for (size_t i = 0; i < input.in.buf.size(); i++) {
        plot[(i*3)+1] = input.in.buf[i];
    }
    return Result::SUCCESS;
}

Result CPU::underlyingPresent() {
    lineVertex->update();
    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
