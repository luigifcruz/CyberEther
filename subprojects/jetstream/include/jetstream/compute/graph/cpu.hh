#ifndef JETSTREAM_COMPUTE_GRAPH_CPU_HH
#define JETSTREAM_COMPUTE_GRAPH_CPU_HH

#include "jetstream/compute/graph/generic.hh"

namespace Jetstream {

class CPU : public Graph {
 public:
    CPU();

    constexpr Device device() const {
        return Device::CPU;
    }

    Result createCompute();
    Result compute();
    Result destroyCompute();
};

}  // namespace Jetstream

#endif
