#ifndef JETSTREAM_MEMORY_CPU_BUFFER_HH
#define JETSTREAM_MEMORY_CPU_BUFFER_HH

#include <cstdlib>
#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/memory/devices/vulkan/buffer.hh"
#endif

namespace Jetstream {

template<>
class TensorBuffer<Device::CPU> {
 public:
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype);

    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          void* ptr);

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer);
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer);
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::CUDA>>& root_buffer);
#endif

    ~TensorBuffer();

    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    constexpr const void* data() const noexcept {
        return buffer;
    }

    constexpr void* data() noexcept {
        return buffer;
    }

 private:
    void* buffer = nullptr;
    bool owns_data = false;
    Device external_memory_device = Device::None;

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    VkDeviceMemory vulkan_memory = VK_NULL_HANDLE;
#endif
};

}  // namespace Jetstream

#endif
