#ifndef JSTCORE_LINEPLOT_CPU_H
#define JSTCORE_LINEPLOT_CPU_H

#include "jstcore/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

class CPU : public Generic  {
public:
    explicit CPU(const Config &, const Input &);

protected:
    Result underlyingCompute() final;
};

} // namespace Jetstream::Lineplot

#endif
