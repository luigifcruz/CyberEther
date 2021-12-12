#ifndef JSTCORE_WATERFALL_CPU_H
#define JSTCORE_WATERFALL_CPU_H

#include "jstcore/waterfall/generic.hpp"

namespace Jetstream::Waterfall {

class CPU : public Generic  {
public:
    explicit CPU(const Config&, const Input&);

protected:
    Result underlyingCompute() final;

    std::vector<float> bin;
};

} // namespace Jetstream::Waterfall

#endif
