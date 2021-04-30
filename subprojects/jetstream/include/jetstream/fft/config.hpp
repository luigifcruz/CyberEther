#ifndef JETSTREAM_FFT_CONFIG_H
#define JETSTREAM_FFT_CONFIG_H

#include "jetstream/base.hpp"

namespace Jetstream::FFT {

struct Config {
    std::shared_ptr<std::vector<std::complex<float>>> input;
    std::shared_ptr<std::vector<std::complex<float>>> output;
};

} // namespace Jetstream::FFT

#endif
