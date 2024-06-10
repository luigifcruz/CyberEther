#ifndef JETSTREAM_BACKEND_BASE_HH
#define JETSTREAM_BACKEND_BASE_HH

#include <unordered_map>
#include <variant>
#include <mutex>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/backend/config.hh"

// TODO: Refactor this entire thing. It's a mess.

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
        std::lock_guard lock(mutex);
        if (!backends.contains(DeviceId)) {
            JST_DEBUG("Initializing {} backend.", DeviceId);
            backends[DeviceId] = std::make_unique<BackendType>(config);
        }
        return Result::SUCCESS;
    }

    Result destroy(const Device& id) {
        std::lock_guard lock(mutex);
        if (backends.contains(id)) {
            JST_DEBUG("Destroying {} backend.", id);
            backends.erase(id);
        }
        return Result::SUCCESS;
    }

    template<Device DeviceId>
    const auto& state() {
        using BackendType = typename GetBackend<DeviceId>::Type;
        if (!backends.contains(DeviceId)) {
            JST_WARN("The {} backend is not initialized. Initializing with default headless settings.", DeviceId);

            Backend::Config config;
            config.headless = true;
            JST_CHECK_THROW(initialize<DeviceId>(config));
        }
        return std::get<std::unique_ptr<BackendType>>(backends[DeviceId]);
    }

    bool initialized(const Device& id) {
        return backends.contains(id);
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
    std::mutex mutex;
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
    return Get().destroy(D);
}

inline Result Destroy(const Device& id) {
    return Get().destroy(id);
}

template<Device D>
bool Initialized() {
    return Get().initialized(D);
}

inline bool Initialized(const Device& id) {
    return Get().initialized(id);
}

inline Result DestroyAll() {
    return Get().destroyAll();
}

}  // namespace Jetstream::Backend

#endif
