#ifndef JETSTREAM_GRAPH_CPU_HH
#define JETSTREAM_GRAPH_CPU_HH

#include "jetstream/graph/generic.hh"

namespace Jetstream {

class CPU : public Graph {
 public:
    CPU();

    const Result createCompute();
    const Result compute();

    constexpr const Device device() const {
        return Device::CPU;
    }
};

}  // namespace Jetstream

#endif
