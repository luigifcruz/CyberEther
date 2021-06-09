#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/base.hpp"

namespace Jetstream::FFT {

using TI = nonstd::span<std::complex<float>>;
using TO = nonstd::span<float>;

struct Config {
    float max_db = 0.0;
    float min_db = -200.0;
    Data<TI> input0;
    Module::Execution policy;
};

class Generic : public Module {
public:
    explicit Generic(Config&);
    virtual ~Generic() = default;

    Config conf() const {
        return cfg;
    }

    Data<TO> output() const {
        return out;
    }

protected:
    Config& cfg;
    Data<TI> in;
    Data<TO> out;

    std::vector<std::complex<float>> window;
};

} // namespace Jetstream::FFT

#endif
