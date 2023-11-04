#ifndef JETSTREAM_MEMORY_VULKAN_BUFFER_HH
#define JETSTREAM_MEMORY_VULKAN_BUFFER_HH

#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

namespace Jetstream {

template<>
class TensorBuffer<Device::Vulkan> {
 public:
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const std::shared_ptr<TensorPrototypeMetadata>& prototype);

    ~TensorBuffer();

    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    const VkDeviceMemory& memory() const noexcept {
        return _memory;
    }

    VkDeviceMemory& memory() noexcept {
        return _memory;
    }

    const VkBuffer& data() const noexcept {
        return _buffer;
    }

    VkBuffer& data() noexcept {
        return _buffer;
    }

 private:
    VkBuffer _buffer;
    VkDeviceMemory _memory;
    bool owns_data = false;
};

}  // namespace Jetstream

#endif
