#ifndef JETSTREAM_COMPUTE_GRAPH_CPU_HH
#define JETSTREAM_COMPUTE_GRAPH_CPU_HH

#include "jetstream/compute/graph/generic.hh"

namespace Jetstream {

class CPU : public Graph {
 public:
    CPU();
    ~CPU();

    constexpr Device device() const {
        return Device::CPU;
    }

    Result create();
    Result compute(std::unordered_set<U64>& yielded);
    Result computeReady();
    Result destroy();
};

}  // namespace Jetstream

#endif
