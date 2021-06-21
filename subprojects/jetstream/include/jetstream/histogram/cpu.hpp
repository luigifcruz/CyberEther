#ifndef JETSTREAM_HST_CPU_H
#define JETSTREAM_HST_CPU_H

#include "jetstream/histogram/generic.hpp"

namespace Jetstream {

class Histogram::CPU : public Histogram  {
public:
    explicit CPU(const Config &);
    ~CPU();

protected:
    Result underlyingCompute() final;
    Result underlyingPresent() final;
};

} // namespace Jetstream

#endif
