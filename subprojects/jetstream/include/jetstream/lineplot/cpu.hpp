#ifndef JETSTREAM_LPT_CPU_H
#define JETSTREAM_LPT_CPU_H

#include "jetstream/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

class CPU : public Generic  {
public:
    using I = cpu::arr::c32;

    explicit CPU(Config&, std::shared_ptr<Module>, I&);
    ~CPU();

protected:
    I& input;

    Result underlyingCompute();
    Result underlyingPresent();
};

} // namespace Jetstream::Lineplot

#endif
