#ifndef JSTCORE_LINEPLOT_CPU_H
#define JSTCORE_LINEPLOT_CPU_H

#include "jstcore/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

template<>
class Backend<Device::CPU> : public Generic  {
public:
    explicit Backend(const Config&, const Input&);

protected:
    const Result underlyingCompute();
};

} // namespace Jetstream::Lineplot

#endif
