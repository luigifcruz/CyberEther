#ifndef JETSTREAM_LPT_CPU_H
#define JETSTREAM_LPT_CPU_H

#include "jetstream/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

class CPU : public Generic  {
public:
    explicit CPU(Config& c);
    ~CPU();

protected:
    Result underlyingCompute();
    Result underlyingPresent();
};

} // namespace Jetstream::Lineplot

#endif

