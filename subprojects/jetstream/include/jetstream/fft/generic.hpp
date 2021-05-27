#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/base.hpp"

namespace Jetstream::FFT {

struct Config {
};

class Generic : public Module {
public:
    explicit Generic(Config& c) : cfg(c) {};
    virtual ~Generic() = default;

    Config conf() const {
        return cfg;
    }

protected:
    Config& cfg;
};

} // namespace Jetstream::FFT

#endif
