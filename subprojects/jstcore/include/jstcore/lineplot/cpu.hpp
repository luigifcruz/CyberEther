#ifndef JETSTREAM_LINEPLOT_CPU_H
#define JETSTREAM_LINEPLOT_CPU_H

#include "jstcore/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

class CPU : public Generic  {
public:
    explicit CPU(const Config &, const Input &);

protected:
    Result underlyingCompute() final;
    Result underlyingPresent() final;
};

} // namespace Jetstream::Lineplot

#endif
