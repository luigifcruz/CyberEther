#ifndef JETSTREAM_COMPUTE_GRAPH_CUDA_HH
#define JETSTREAM_COMPUTE_GRAPH_CUDA_HH

#include "jetstream/compute/graph/generic.hh"

namespace Jetstream {

class CUDA : public Graph {
 public:
    CUDA();

    constexpr Device device() const {
        return Device::CUDA;
    }

    Result create();
    Result compute();
    Result computeReady();
    Result destroy();

 private:
    cudaStream_t stream;
};

}  // namespace Jetstream

#endif
