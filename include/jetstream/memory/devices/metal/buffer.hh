#ifndef JETSTREAM_MEMORY_METAL_BUFFER_HH
#define JETSTREAM_MEMORY_METAL_BUFFER_HH

#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

namespace Jetstream {

template<>
class TensorBuffer<Device::Metal> {
 public:
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype);

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer);
    static bool CanImport(const TensorBuffer<Device::CPU>& root_buffer) noexcept;
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

    constexpr const bool& host_accessible() const {
        return _host_accessible;
    }

    constexpr const bool& device_native() const {
        return _device_native;
    }

    constexpr const bool& host_native() const {
        return _host_native;
    }

    constexpr const MTL::Buffer* data() const noexcept {
        return buffer;
    }

    constexpr MTL::Buffer* data() noexcept {
        return buffer;
    }

 private:
    MTL::Buffer* buffer;
    bool owns_data = false;
    bool _host_accessible = false;
    bool _device_native = false;
    bool _host_native = false;
};

}  // namespace Jetstream

#endif
