#ifndef JETSTREAM_MEMORY_CPU_BUFFER_HH
#define JETSTREAM_MEMORY_CPU_BUFFER_HH

#include <cstdlib>
#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

namespace Jetstream {

template<>
class TensorBuffer<Device::CPU> {
 public:
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const std::shared_ptr<TensorPrototypeMetadata>& prototype);

    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const std::shared_ptr<TensorPrototypeMetadata>& prototype,
                          void* ptr);

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const std::shared_ptr<TensorPrototypeMetadata>& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer);
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const std::shared_ptr<TensorPrototypeMetadata>& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer);
#endif

    ~TensorBuffer();

    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    const void* data() const noexcept {
        return buffer;
    }

    void* data() noexcept {
        return buffer;
    }

 private:
    void* buffer = nullptr;
    bool owns_data = false;
};

}  // namespace Jetstream

#endif
