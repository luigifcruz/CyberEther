#ifndef JETSTREAM_MEMORY_CPU_BUFFER_HH
#define JETSTREAM_MEMORY_CPU_BUFFER_HH

#include <cstdlib>
#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

namespace Jetstream {

template<>
class TensorBuffer<Device::CPU> : public TensorBufferBase {
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
    static bool CanImport(const TensorBuffer<Device::Metal>& root_buffer) noexcept;
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer);
    static bool CanImport(const TensorBuffer<Device::Vulkan>& root_buffer) noexcept;
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::CUDA>>& root_buffer);
    static bool CanImport(const TensorBuffer<Device::CUDA>& root_buffer) noexcept;
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

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    VkDeviceMemory vulkan_memory = VK_NULL_HANDLE;
#endif
};

}  // namespace Jetstream

#endif
