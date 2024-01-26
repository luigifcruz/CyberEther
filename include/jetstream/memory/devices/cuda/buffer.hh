#ifndef JETSTREAM_MEMORY_CUDA_BUFFER_HH
#define JETSTREAM_MEMORY_CUDA_BUFFER_HH

#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/memory/devices/vulkan/buffer.hh"
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/memory/devices/cpu/buffer.hh"
#endif

namespace Jetstream {

template<>
class TensorBuffer<Device::CUDA> {
 public:
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const bool& host_accessible = false);

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer);
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer);
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer);
#endif

    ~TensorBuffer();

    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    constexpr const bool& host_accessible() const {
        return _host_accessible;
    }

    constexpr const bool& device_native() const {
        return _device_native;
    }

    constexpr const bool& host_native() const {
        return _host_native;
    }

    constexpr const void* data() const noexcept {
        return buffer;
    }

    constexpr void* data() noexcept {
        return buffer;
    }

 private:
    void* buffer;
    bool owns_data = false;
    bool _host_accessible = false;
    bool _device_native = false;
    bool _host_native = false;
    Device external_memory_device = Device::None;

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    int vulkan_file_descriptor = 0;
    CUexternalMemory vulkan_external_memory = nullptr;
#endif
};

}  // namespace Jetstream

#endif
