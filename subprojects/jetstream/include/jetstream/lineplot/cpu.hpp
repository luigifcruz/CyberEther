#ifndef JETSTREAM_LPT_CPU_H
#define JETSTREAM_LPT_CPU_H

#include "jetstream/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

using I = cpu::arr::c32;

class CPU : public Generic  {
public:
    explicit CPU(Config&, I&);
    ~CPU();

protected:
    I& input;

    Result underlyingCompute();
    Result underlyingPresent();
};

} // namespace Jetstream::Lineplot

#endif
