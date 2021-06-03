#ifndef JETSTREAM_WTF_CPU_H
#define JETSTREAM_WTF_CPU_H

#include "jetstream/waterfall/generic.hpp"

namespace Jetstream::Waterfall {

class CPU : public Generic  {
public:
    explicit CPU(Config&);
    ~CPU();

protected:
    Result underlyingCompute();
    Result underlyingPresent();
};

} // namespace Jetstream::Waterfall

#endif
