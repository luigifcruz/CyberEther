#ifndef JETSTREAM_HST_CPU_H
#define JETSTREAM_HST_CPU_H

#include "jetstream/histogram/generic.hpp"

namespace Jetstream::Histogram {

class CPU : public Generic  {
public:
    explicit CPU(Config&);
    ~CPU();

protected:
    Result underlyingCompute();
    Result underlyingPresent();
};

} // namespace Jetstream::Histogram

#endif
