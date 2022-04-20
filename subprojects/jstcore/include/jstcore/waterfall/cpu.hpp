#ifndef JSTCORE_WATERFALL_CPU_H
#define JSTCORE_WATERFALL_CPU_H

#include "jstcore/waterfall/generic.hpp"

namespace Jetstream::Waterfall {

template<>
class Backend<Device::CPU> : public Generic  {
public:
    explicit Backend(const Config&, const Input&);

protected:
    const Result underlyingCompute();

    std::vector<float> bin;
};

} // namespace Jetstream::Waterfall

#endif
