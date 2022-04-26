#include "jetstream/modules/fft/base.hh"

namespace Jetstream {

template<>
FFT<Device::CPU>::FFT(const Config& config, const Input& input) 
    : config(config), input(input) {
    std::cout << "FFT CPU" << std::endl;
}

template<>
const Result FFT<Device::CPU>::compute() {
    std::cout << "FFT CPU Compute" << std::endl;
    return Result::SUCCESS;
}
    
}  // namespace Jetstream
