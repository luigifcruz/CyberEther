#ifndef JETSTREAM_BACKEND_BASE_HH
#define JETSTREAM_BACKEND_BASE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/backend/config.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/backend/devices/cpu/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/devices/metal/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/backend/devices/vulkan/base.hh"
#endif

namespace Jetstream::Backend {

class JETSTREAM_API Instance {
 public:
    template<Device D>
    const Result initialize(const Config& config);

    template<Device D>
    const Result destroy();

    template<Device D>
    const auto& state();

 private:
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    std::unique_ptr<CPU> cpu;
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    std::unique_ptr<Metal> metal;
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    std::unique_ptr<Vulkan> vulkan;
#endif
};

Instance& Get();

template<Device D>
const auto& State() {
    return Get().state<D>();
}

template<Device D>
Result Initialize(const Config& config) {
    return Get().initialize<D>(config);
}

template<Device D>
Result Destroy() {
    return Get().destroy<D>();
}

}  // namespace Jetstream::Backend

#endif
