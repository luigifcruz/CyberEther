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

#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
#include "jetstream/backend/devices/webgpu/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/backend/devices/cpu/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/backend/devices/cuda/base.hh"
#endif

namespace Jetstream::Backend {

template<Device DeviceId>
struct GetBackend {
    static constexpr bool enabled = false;
};

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
template<>
struct GetBackend<Device::Metal> {
    static constexpr bool enabled = true;
    using Type = Metal;  
};
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
template<>
struct GetBackend<Device::Vulkan> {
    static constexpr bool enabled = true;
    using Type = Vulkan;  
};
#endif

#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
template<>
struct GetBackend<Device::WebGPU> {
    static constexpr bool enabled = true;
    using Type = WebGPU;
};
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
template<>
struct GetBackend<Device::CPU> {
    static constexpr bool enabled = true;
    using Type = CPU;  
};
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
template<>
struct GetBackend<Device::CUDA> {
    static constexpr bool enabled = true;
    using Type = CUDA;  
};
#endif

class JETSTREAM_API Instance {
 public:
    template<Device DeviceId>
    Result initialize(const Config& config) {
        using BackendType = typename GetBackend<DeviceId>::Type;
        if (!backends.contains(DeviceId)) {
            JST_DEBUG("Initializing {} backend.", DeviceId);
            backends[DeviceId] = std::make_unique<BackendType>(config);
        }
        return Result::SUCCESS;
    }

    template<Device DeviceId>
    Result destroy() {
        if (backends.contains(DeviceId)) {
            JST_DEBUG("Destroying {} backend.", DeviceId);
            backends.erase(DeviceId);
        }
        return Result::SUCCESS;
    }

    template<Device DeviceId>
    const auto& state() {
        using BackendType = typename GetBackend<DeviceId>::Type;
        if (!backends.contains(DeviceId)) {
            JST_WARN("The {} backend is not initialized. Initializing with default settings.", DeviceId);
            JST_CHECK_THROW(initialize<DeviceId>({}));
        }
        return std::get<std::unique_ptr<BackendType>>(backends[DeviceId]);
    }

    Result destroyAll() {
        backends.clear();
        return Result::SUCCESS;
    }

 private:
    typedef std::variant<
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        std::unique_ptr<Metal>,
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        std::unique_ptr<Vulkan>,
#endif
#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
        std::unique_ptr<WebGPU>,
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        std::unique_ptr<CUDA>,
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

inline Result DestroyAll() {
    return Get().destroyAll();
}

}  // namespace Jetstream::Backend

#endif
