#include "jetstream/lineplot/cpu.hpp"

namespace Jetstream::Lineplot {

CPU::CPU(Config& c) : Generic(c) {
    plotVbo.data = plot.data();
    JETSTREAM_CHECK_THROW(this->_initRender());
}

CPU::~CPU() {
}

Result CPU::_compute() {
    for (int i = 0; i < in.buf.size(); i++) {
        plot[(i*3)+1] = in.buf[i];
    }

    return Result::SUCCESS;
}

Result CPU::_present() {
    lineVertex->update();

    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
