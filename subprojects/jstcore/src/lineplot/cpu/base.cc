#include "jstcore/lineplot/cpu.hpp"

namespace Jetstream::Lineplot {

Backend<Device::CPU>::Backend(const Config& config, const Input& input) : Generic(config, input) {
    if ((input.in.location & Locale::CPU) != Locale::CPU) {
        std::cerr << "[LINEPLOT::CPU] This implementation expects a Locale::CPU input." << std::endl;
        JST_CHECK_THROW(Result::ERROR);
    }

    this->initRender(plot.data());
}

const Result Backend<Device::CPU>::underlyingCompute() {
    for (size_t i = 0; i < input.in.buf.size(); i++) {
        plot[(i*3)+1] = input.in.buf[i];
    }
    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
