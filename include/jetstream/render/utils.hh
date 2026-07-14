#ifndef JETSTREAM_RENDER_UTILS_HH
#define JETSTREAM_RENDER_UTILS_HH

#include "jetstream/render/base.hh"
#include "jetstream/memory/tensor.hh"

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/backend/devices/cuda/base.hh"
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/backend/devices/vulkan/base.hh"
#include "jetstream/memory/devices/vulkan/buffer.hh"
#endif

namespace Jetstream {

inline bool CanUseRenderZeroCopy(const auto& window, const DeviceType tensorDevice) {
    if (!window) {
        return false;
    }

    const auto renderDevice = window->device();
    if (renderDevice == tensorDevice) {
        return true;
    }

    if (tensorDevice == DeviceType::CPU && renderDevice == DeviceType::Metal) {
        return true;
    }

#if defined(JETSTREAM_BACKEND_CUDA_AVAILABLE) && defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE)
    if (tensorDevice == DeviceType::CUDA && renderDevice == DeviceType::Vulkan) {
        return Backend::State<DeviceType::CUDA>()->canExportDeviceMemory() &&
               Backend::State<DeviceType::Vulkan>()->canImportDeviceMemory();
    }
#endif

    return false;
}

inline Result ConvertToOptimalStorage(auto& window,
                                      Tensor& tensor,
                                      Tensor& storage) {
    const auto renderDevice = window->device();

    if (renderDevice == tensor.device()) {
        storage = tensor.clone();

        JST_TRACE("[RENDER] Tensor Device: {} | Render Device: {} | Optimal Device: {} | Zero-Copy: {}",
                  tensor.device(), renderDevice, storage.device(), "YES");

        return Result::SUCCESS;
    }

    if (CanUseRenderZeroCopy(window, tensor.device())) {
        JST_CHECK(storage.create(renderDevice, tensor));

        JST_TRACE("[RENDER] Tensor Device: {} | Render Device: {} | Optimal Device: {} | Zero-Copy: {}",
                  tensor.device(), renderDevice, storage.device(), "YES");

        return Result::SUCCESS;
    }

    if (tensor.device() == DeviceType::CPU) {
        storage = tensor.clone();
    } else {
        JST_CHECK(storage.create(DeviceType::CPU, tensor));
    }

    JST_TRACE("[RENDER] Tensor Device: {} | Render Device: {} | Optimal Device: {} | Zero-Copy: {}",
              tensor.device(), renderDevice, storage.device(), "NO");

    return Result::SUCCESS;
}

inline void* RenderStorageBuffer(Tensor& storage) {
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (storage.device() == DeviceType::Vulkan) {
        auto* backend = static_cast<VulkanBufferBackend*>(storage.buffer().backend());
        return backend ? reinterpret_cast<void*>(backend->buffer()) : nullptr;
    }
#endif

    return storage.data();
}

}  // namespace Jetstream

#endif
