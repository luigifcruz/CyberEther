#ifndef JETSTREAM_GRAPH_BASE_HH
#define JETSTREAM_GRAPH_BASE_HH

#include "jetstream/graph/generic.hh"

#include "jetstream/graph/cpu.hh"
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
#include "jetstream/graph/metal.hh"
#endif

namespace Jetstream {

inline std::unique_ptr<Graph> NewGraph(const Device& device) {
    switch (device) {
        case Device::None:
            return nullptr;
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE 
        case Device::CPU:
            return std::make_unique<CPU>();
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE 
        case Device::CUDA:
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE 
        case Device::Vulkan:
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE 
        case Device::Metal:
            return std::make_unique<Metal>();
#endif
    };
}

}  // namespace Jetstream

#endif
