#ifndef JETSTREAM_BACKEND_BASE_HH
#define JETSTREAM_BACKEND_BASE_HH

#include <unordered_map>
#include <variant>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/backend/config.hh"

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/devices/metal/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/backend/devices/vulkan/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/backend/devices/cpu/base.hh"
#endif

namespace Jetstream::Backend {

template<Device DeviceId>
struct GetBackend {};

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
template<>
struct GetBackend<Device::Metal> {
    using Type = Metal;  
};
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
template<>
struct GetBackend<Device::Vulkan> {
    using Type = Vulkan;  
};
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
template<>
struct GetBackend<Device::CPU> {
    using Type = CPU;  
};
#endif

class JETSTREAM_API Instance {
 public:
    template<Device DeviceId>
    const Result initialize(const Config& config) {
        using BackendType = typename GetBackend<DeviceId>::Type;
        if (!backends.count(DeviceId)) {
            JST_DEBUG("Initializing {} backend.", DeviceId);
            backends[DeviceId] = std::make_unique<BackendType>(config);
        }
        return Result::SUCCESS;
    }

    template<Device DeviceId>
    const Result destroy() {
        if (backends.count(DeviceId)) {
            JST_DEBUG("Destroying {} backend.", DeviceId);
            backends.erase(DeviceId);
        }
        return Result::SUCCESS;
    }

    template<Device DeviceId>
    const auto& state() {
        using BackendType = typename GetBackend<DeviceId>::Type;
        if (!backends.count(DeviceId)) {
            JST_DEBUG("The {} backend is not initialized.", DeviceId);
            JST_CHECK_THROW(Result::ERROR);
        }
        return std::get<std::unique_ptr<BackendType>>(backends[DeviceId]);
    }

    ~Instance() {
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        destroy<Device::Metal>();
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        destroy<Device::Vulkan>();
#endif
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        destroy<Device::CPU>();
#endif
        
    }

 private:
    typedef std::variant<
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        std::unique_ptr<Metal>,
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        std::unique_ptr<Vulkan>,
#endif
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        std::unique_ptr<CPU>
#endif
    > BackendHolder;

    std::unordered_map<Device, BackendHolder> backends;
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
