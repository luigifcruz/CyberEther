#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/base.hpp"

namespace Jetstream::FFT {

using T = std::vector<std::complex<float>>;

struct Config {
    Data<T> input0;
    Module::Execution policy;
};

class Generic : public Module {
public:
    explicit Generic(Config& c)
        : Module(c.policy),
          cfg(c),
          in(c.input0) {
    }
    virtual ~Generic() = default;

    Config conf() const {
        return cfg;
    }

    Data<T> output() const {
        return out;
    }

protected:
    Config& cfg;
    Data<T> in;
    Data<T> out;
};

} // namespace Jetstream::FFT

#endif
