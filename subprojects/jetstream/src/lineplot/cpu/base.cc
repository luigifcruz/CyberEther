#include "jetstream/lineplot/cpu.hpp"

namespace Jetstream {

Lineplot::CPU::CPU(const Config & c, Manifest & i) : Lineplot(c, i) {
    JETSTREAM_CHECK_THROW(this->_initRender(plot.data()));
}

Lineplot::CPU::~CPU() {
}

Result Lineplot::CPU::_compute() {
    for (int i = 0; i < in.buf.size(); i++) {
        plot[(i*3)+1] = in.buf[i];
    }

    return Result::SUCCESS;
}

Result Lineplot::CPU::_present() {
    lineVertex->update();

    return Result::SUCCESS;
}

} // namespace Jetstream
