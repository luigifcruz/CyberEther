#include "jstcore/waterfall/cpu.hpp"

namespace Jetstream::Waterfall {

CPU::CPU(const Config& config, const Input& input) : Generic(config, input) {
    if ((input.in.location & Locale::CPU) != Locale::CPU) {
        std::cerr << "[WATERFALL::CPU] This implementation expects a Locale::CPU input." << std::endl;
        JST_CHECK_THROW(Result::ERROR);
    }

    ymax = config.size.height;
    bin.resize(input.in.buf.size() * ymax);
    this->initRender((uint8_t*)bin.data());
}

const Result CPU::underlyingCompute() {
    std::copy(input.in.buf.begin(), input.in.buf.end(), bin.begin()+(inc * input.in.buf.size()));
    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
