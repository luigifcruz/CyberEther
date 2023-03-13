#ifndef JETSTREAM_GRAPH_METAL_HH
#define JETSTREAM_GRAPH_METAL_HH

#include "jetstream/graph/generic.hh"

namespace Jetstream {

class Metal : public Graph {
 public:
    Metal();

    const Result createCompute();
    const Result compute();

    constexpr const Device device() const {
        return Device::Metal;
    }

 private:
    MTL::CommandQueue* commandQueue;
};

}  // namespace Jetstream

#endif
