#ifndef JETSTREAM_COMPUTE_GRAPH_BASE_HH
#define JETSTREAM_COMPUTE_GRAPH_BASE_HH

#include "jetstream/compute/graph/generic.hh"

#ifdef JETSTREAM_GRAPH_CPU_AVAILABLE
#include "jetstream/compute/graph/cpu.hh"
#endif
#ifdef JETSTREAM_GRAPH_METAL_AVAILABLE
#include "jetstream/compute/graph/metal.hh"
#endif

namespace Jetstream {

inline std::unique_ptr<Graph> NewGraph(const Device& device) {
    switch (device) {
#ifdef JETSTREAM_GRAPH_CPU_AVAILABLE 
        case Device::CPU:
            return std::make_unique<CPU>();
#endif
// #ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE 
//         case Device::CUDA:
//             return std::make_unique<CUDA>();
// #endif
// #ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE 
//         case Device::Vulkan:
//             return std::make_unique<Vulkan>();
// #endif
#ifdef JETSTREAM_GRAPH_METAL_AVAILABLE 
        case Device::Metal:
            return std::make_unique<Metal>();
#endif
        default:
            JST_ERROR("[GRAPH] Backend not supported yet.");
            throw Result::ERROR;
    };
}

}  // namespace Jetstream

#endif
