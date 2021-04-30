#ifndef JETSTREAM_LPT_CPU_H
#define JETSTREAM_LPT_CPU_H

#include "jetstream/lineplot/config.hpp"

namespace Jetstream::Lineplot {

class CPU : public Transform, public State {
public:
    explicit CPU(Config& c);
    ~CPU();

protected:
    Config& cfg;

    Result underlyingCompute();
    Result underlyingPresent();
};

} // namespace Jetstream::Lineplot

#endif

